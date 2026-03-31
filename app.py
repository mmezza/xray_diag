import uuid
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config import settings
from services.claude_service import ClaudeService
from services.elasticsearch_service import ElasticsearchService
from services.embedding_service import EmbeddingService

app = FastAPI(
    title="XRay Diagnostics",
    description="Análise semântica de raios-X com IA + Vector Search",
    version="2.0.0",
)

app.mount("/static",  StaticFiles(directory="static"),         name="static")
app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")
templates = Jinja2Templates(directory="templates")

vision_service    = ClaudeService()
es_service        = ElasticsearchService()
embedding_service = EmbeddingService()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".pdf"}
MAX_SIZE_BYTES     = settings.max_file_size_mb * 1024 * 1024

Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)


class SearchRequest(BaseModel):
    query: str
    size:  int = 10


@app.on_event("startup")
async def startup_event():
    await es_service.initialize()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ── upload + analyse ──────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_and_analyze(file: UploadFile = File(...)):
    """
    1. Save the uploaded image.
    2. Analyse it with GPT-4o vision → structured diagnosis JSON.
    3. Embed the diagnostic text with text-embedding-3-small.
    4. Store doc + vector in Elasticsearch (or memory).
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo não suportado. Use: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Arquivo muito grande. Máximo: {settings.max_file_size_mb} MB",
        )

    image_id = str(uuid.uuid4())
    filename = f"{image_id}{suffix}"
    filepath = Path(settings.upload_dir) / filename
    filepath.write_bytes(content)

    media_type = _get_media_type(suffix)
    image_b64  = base64.standard_b64encode(content).decode("utf-8")

    # ── step 1: vision analysis ──
    try:
        diagnosis = await vision_service.analyze_xray(
            image_b64=image_b64,
            media_type=media_type,
            original_filename=file.filename,
        )
    except Exception as exc:
        filepath.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Erro na análise: {exc}")

    # ── step 2: generate embedding (non-blocking; skip if it fails) ──
    embedding: Optional[list[float]] = None
    try:
        embedding = await embedding_service.embed_diagnosis(diagnosis)
    except Exception as exc:
        # Vector search won't work for this doc, but the rest still works
        import logging
        logging.getLogger(__name__).warning(f"Embedding falhou: {exc}")

    # ── step 3: persist ──
    doc = {
        "id":                image_id,
        "original_filename": file.filename,
        "stored_filename":   filename,
        "media_type":        media_type,
        "upload_date":       datetime.utcnow().isoformat(),
        "file_size_bytes":   len(content),
        "diagnosis":         diagnosis,
        "has_embedding":     embedding is not None,
    }
    await es_service.store_document(image_id, doc, embedding=embedding)

    return JSONResponse({
        "id":             image_id,
        "filename":       file.filename,
        "stored_filename": filename,
        "upload_date":    doc["upload_date"],
        "diagnosis":      diagnosis,
        "has_embedding":  doc["has_embedding"],
    })


# ── CRUD ──────────────────────────────────────────────────────────────────

@app.get("/api/images")
async def list_images(size: int = 20):
    results = await es_service.list_documents(size=size)
    return {"images": results, "total": len(results)}


@app.get("/api/images/{image_id}")
async def get_image(image_id: str):
    doc = await es_service.get_document(image_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Imagem não encontrada")
    return doc


@app.delete("/api/images/{image_id}")
async def delete_image(image_id: str):
    doc = await es_service.get_document(image_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Imagem não encontrada")
    (Path(settings.upload_dir) / doc.get("stored_filename", "")).unlink(missing_ok=True)
    await es_service.delete_document(image_id)
    return {"message": "Imagem removida com sucesso"}


# ── vector search ─────────────────────────────────────────────────────────

@app.post("/api/search")
async def search_images(req: SearchRequest):
    """
    Hybrid semantic search: embeds the query text and runs
    kNN (60%) + BM25 (40%) on Elasticsearch.
    Falls back to cosine similarity on in-memory vectors.
    """
    query_embedding: Optional[list[float]] = None
    try:
        query_embedding = await embedding_service.embed(req.query)
    except Exception:
        pass  # graceful degradation to BM25-only

    results = await es_service.search(
        query=req.query,
        query_embedding=query_embedding,
        size=req.size,
    )
    return {
        "results":      results,
        "total":        len(results),
        "query":        req.query,
        "vector_used":  query_embedding is not None,
    }


@app.get("/api/images/{image_id}/similar")
async def find_similar(image_id: str, size: int = 5):
    """
    Find the `size` most similar X-rays to `image_id`
    using pure kNN cosine similarity on diagnosis embeddings.
    """
    doc = await es_service.get_document(image_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Imagem não encontrada")

    similar = await es_service.find_similar(image_id, size=size)
    return {
        "source_id": image_id,
        "similar":   similar,
        "total":     len(similar),
    }


# ── stats / health ────────────────────────────────────────────────────────

@app.get("/api/stats")
async def get_stats():
    return await es_service.get_stats()


@app.get("/api/health")
async def health():
    es_ok = await es_service.ping()
    return {
        "status":        "ok",
        "openai":        bool(settings.openai_api_key),
        "elasticsearch": es_ok,
        "storage_mode":  "elasticsearch" if es_ok else "memory",
        "vector_search": True,
    }


# ── util ──────────────────────────────────────────────────────────────────

def _get_media_type(suffix: str) -> str:
    return {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".webp": "image/webp",
        ".gif":  "image/gif",
        ".bmp":  "image/bmp",
        ".tiff": "image/tiff",
        ".pdf":  "application/pdf",
    }.get(suffix, "image/jpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=settings.app_host, port=settings.app_port, reload=True)

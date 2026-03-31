import os
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

app = FastAPI(
    title="XRay Diagnostics",
    description="Análise semântica de raios-X com IA",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")
templates = Jinja2Templates(directory="templates")

claude_service = ClaudeService()
es_service = ElasticsearchService()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".pdf"}
MAX_SIZE_BYTES = settings.max_file_size_mb * 1024 * 1024

Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)


class SearchRequest(BaseModel):
    query: str
    size: int = 10


@app.on_event("startup")
async def startup_event():
    await es_service.initialize()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload")
async def upload_and_analyze(file: UploadFile = File(...)):
    """Upload a X-ray image and analyze it with Claude."""
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
            detail=f"Arquivo muito grande. Máximo: {settings.max_file_size_mb}MB",
        )

    image_id = str(uuid.uuid4())
    filename = f"{image_id}{suffix}"
    filepath = Path(settings.upload_dir) / filename
    with open(filepath, "wb") as f:
        f.write(content)

    media_type = _get_media_type(suffix)
    image_b64 = base64.standard_b64encode(content).decode("utf-8")

    try:
        diagnosis = await claude_service.analyze_xray(
            image_b64=image_b64,
            media_type=media_type,
            original_filename=file.filename,
        )
    except Exception as e:
        filepath.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Erro na análise: {str(e)}")

    doc = {
        "id": image_id,
        "original_filename": file.filename,
        "stored_filename": filename,
        "media_type": media_type,
        "upload_date": datetime.utcnow().isoformat(),
        "file_size_bytes": len(content),
        "diagnosis": diagnosis,
    }

    await es_service.store_document(image_id, doc)

    return JSONResponse(
        {
            "id": image_id,
            "filename": file.filename,
            "stored_filename": filename,
            "upload_date": doc["upload_date"],
            "diagnosis": diagnosis,
        }
    )


@app.get("/api/images")
async def list_images(size: int = 20):
    """List all analyzed images."""
    results = await es_service.list_documents(size=size)
    return {"images": results, "total": len(results)}


@app.get("/api/images/{image_id}")
async def get_image(image_id: str):
    """Get a specific image analysis."""
    doc = await es_service.get_document(image_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Imagem não encontrada")
    return doc


@app.delete("/api/images/{image_id}")
async def delete_image(image_id: str):
    """Delete an image and its analysis."""
    doc = await es_service.get_document(image_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Imagem não encontrada")

    filepath = Path(settings.upload_dir) / doc.get("stored_filename", "")
    filepath.unlink(missing_ok=True)

    await es_service.delete_document(image_id)
    return {"message": "Imagem removida com sucesso"}


@app.post("/api/search")
async def search_images(req: SearchRequest):
    """Semantic search across X-ray analyses."""
    results = await es_service.search(query=req.query, size=req.size)
    return {"results": results, "total": len(results), "query": req.query}


@app.get("/api/stats")
async def get_stats():
    """Return statistics about stored analyses."""
    stats = await es_service.get_stats()
    return stats


@app.get("/api/health")
async def health():
    es_ok = await es_service.ping()
    return {
        "status": "ok",
        "openai": bool(settings.openai_api_key),
        "elasticsearch": es_ok,
        "storage_mode": "elasticsearch" if es_ok else "memory",
    }


def _get_media_type(suffix: str) -> str:
    mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".pdf": "application/pdf",
    }
    return mapping.get(suffix, "image/jpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=settings.app_host, port=settings.app_port, reload=True)

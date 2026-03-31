"""
Embedding service using OpenAI text-embedding-3-small.

Converts diagnostic text into 1536-dimensional vectors for
semantic / kNN search in Elasticsearch.
"""

from openai import AsyncOpenAI
from config import settings

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS  = 1536


class EmbeddingService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def embed(self, text: str) -> list[float]:
        """Return a 1536-d embedding vector for the given text."""
        text = text.strip().replace("\n", " ")
        response = await self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    async def embed_diagnosis(self, diagnosis: dict) -> list[float]:
        """Build a rich text from a diagnosis dict and embed it."""
        return await self.embed(build_embedding_text(diagnosis))


def build_embedding_text(diagnosis: dict) -> str:
    """
    Create a semantically rich text from a structured diagnosis.
    This is what gets embedded and compared for vector similarity.
    """
    parts = [
        f"Região: {diagnosis.get('region', '')}",
        f"Risco: {diagnosis.get('risk_level', '')}",
        f"Resumo: {diagnosis.get('summary', '')}",
    ]
    for f in diagnosis.get("findings", []):
        loc  = f.get("location", "")
        desc = f.get("description", "")
        sev  = f.get("severity", "")
        parts.append(f"Achado em {loc}: {desc} (severidade: {sev})")
    for c in diagnosis.get("areas_of_concern", []):
        area = c.get("area", "")
        obs  = c.get("observation", "")
        parts.append(f"Área de atenção — {area}: {obs}")
    for s in diagnosis.get("normal_structures", []):
        parts.append(f"Normal: {s}")
    return " | ".join(p for p in parts if p.strip(" |:"))

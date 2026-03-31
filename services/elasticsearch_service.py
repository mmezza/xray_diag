"""
Elasticsearch service with in-memory fallback.

When ES credentials are not configured, all data is stored in memory
(lost on restart). Configure ELASTICSEARCH_URL + ELASTICSEARCH_API_KEY
(or username/password) in .env to enable persistent storage and vector search.

Index mapping uses:
  - 'findings_text' keyword analyzed field for full-text search
  - 'risk_level' keyword field for aggregations / filtering
  - 'upload_date' date field for sorting
  - 'embedding' dense_vector (768d) reserved for future semantic embeddings
"""

import logging
from datetime import datetime
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

INDEX_NAME = "xray_diagnostics"

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "original_filename": {"type": "keyword"},
            "stored_filename": {"type": "keyword"},
            "media_type": {"type": "keyword"},
            "upload_date": {"type": "date"},
            "file_size_bytes": {"type": "long"},
            "region": {"type": "keyword"},
            "risk_level": {"type": "keyword"},
            "confidence": {"type": "float"},
            "summary": {"type": "text", "analyzer": "portuguese"},
            "findings_text": {"type": "text", "analyzer": "portuguese"},
            "diagnosis": {"type": "object", "enabled": True},
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "portuguese": {
                    "tokenizer": "standard",
                    "filter": ["lowercase", "portuguese_stop", "portuguese_stem"],
                }
            },
            "filter": {
                "portuguese_stop": {
                    "type": "stop",
                    "stopwords": "_portuguese_",
                },
                "portuguese_stem": {"type": "stemmer", "language": "portuguese"},
            },
        },
    },
}


class ElasticsearchService:
    """
    Wraps Elasticsearch operations.
    Falls back to in-memory dict store when ES is not configured.
    """

    def __init__(self):
        self._es = None
        self._memory_store: dict[str, dict] = {}
        self._es_available = False

    async def initialize(self):
        if not settings.elasticsearch_url:
            logger.info("Elasticsearch não configurado — usando armazenamento em memória.")
            return

        try:
            from elasticsearch import AsyncElasticsearch

            kwargs = {"hosts": [settings.elasticsearch_url]}

            if settings.elasticsearch_api_key:
                kwargs["api_key"] = settings.elasticsearch_api_key
            elif settings.elasticsearch_username and settings.elasticsearch_password:
                kwargs["basic_auth"] = (
                    settings.elasticsearch_username,
                    settings.elasticsearch_password,
                )

            self._es = AsyncElasticsearch(**kwargs)

            if not await self._es.ping():
                logger.warning("Elasticsearch não respondeu ao ping — usando memória.")
                self._es = None
                return

            await self._ensure_index()
            self._es_available = True
            logger.info("Elasticsearch conectado com sucesso.")

        except Exception as exc:
            logger.warning(f"Falha ao conectar ao Elasticsearch: {exc} — usando memória.")
            self._es = None

    async def _ensure_index(self):
        exists = await self._es.indices.exists(index=INDEX_NAME)
        if not exists:
            await self._es.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)
            logger.info(f"Índice '{INDEX_NAME}' criado.")

    async def ping(self) -> bool:
        if self._es:
            try:
                return await self._es.ping()
            except Exception:
                return False
        return False

    async def store_document(self, doc_id: str, doc: dict):
        flat = _flatten_for_index(doc)
        if self._es_available and self._es:
            await self._es.index(index=INDEX_NAME, id=doc_id, document=flat)
        else:
            self._memory_store[doc_id] = doc

    async def get_document(self, doc_id: str) -> Optional[dict]:
        if self._es_available and self._es:
            try:
                res = await self._es.get(index=INDEX_NAME, id=doc_id)
                return res["_source"]
            except Exception:
                return None
        return self._memory_store.get(doc_id)

    async def delete_document(self, doc_id: str):
        if self._es_available and self._es:
            try:
                await self._es.delete(index=INDEX_NAME, id=doc_id)
            except Exception:
                pass
        else:
            self._memory_store.pop(doc_id, None)

    async def list_documents(self, size: int = 20) -> list[dict]:
        if self._es_available and self._es:
            res = await self._es.search(
                index=INDEX_NAME,
                body={
                    "query": {"match_all": {}},
                    "sort": [{"upload_date": {"order": "desc"}}],
                    "size": size,
                },
            )
            return [hit["_source"] for hit in res["hits"]["hits"]]
        else:
            docs = list(self._memory_store.values())
            docs.sort(key=lambda d: d.get("upload_date", ""), reverse=True)
            return docs[:size]

    async def search(self, query: str, size: int = 10) -> list[dict]:
        if self._es_available and self._es:
            body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": [
                            "summary^3",
                            "findings_text^2",
                            "region",
                            "risk_level",
                            "original_filename",
                        ],
                        "type": "best_fields",
                        "fuzziness": "AUTO",
                    }
                },
                "size": size,
                "highlight": {
                    "fields": {
                        "summary": {},
                        "findings_text": {},
                    }
                },
            }
            res = await self._es.search(index=INDEX_NAME, body=body)
            results = []
            for hit in res["hits"]["hits"]:
                doc = hit["_source"]
                doc["_score"] = hit["_score"]
                doc["_highlights"] = hit.get("highlight", {})
                results.append(doc)
            return results
        else:
            q = query.lower()
            results = []
            for doc in self._memory_store.values():
                searchable = _build_search_text(doc).lower()
                if q in searchable:
                    results.append(doc)
            return results[:size]

    async def get_stats(self) -> dict:
        if self._es_available and self._es:
            agg_res = await self._es.search(
                index=INDEX_NAME,
                body={
                    "size": 0,
                    "aggs": {
                        "by_risk": {"terms": {"field": "risk_level"}},
                        "by_region": {"terms": {"field": "region"}},
                        "total": {"value_count": {"field": "id"}},
                    },
                },
            )
            aggs = agg_res.get("aggregations", {})
            return {
                "total": agg_res["hits"]["total"]["value"],
                "by_risk": {
                    b["key"]: b["doc_count"]
                    for b in aggs.get("by_risk", {}).get("buckets", [])
                },
                "by_region": {
                    b["key"]: b["doc_count"]
                    for b in aggs.get("by_region", {}).get("buckets", [])
                },
                "storage": "elasticsearch",
            }
        else:
            docs = list(self._memory_store.values())
            by_risk: dict[str, int] = {}
            by_region: dict[str, int] = {}
            for doc in docs:
                diag = doc.get("diagnosis", {})
                r = diag.get("risk_level", "Desconhecido")
                reg = diag.get("region", "Desconhecido")
                by_risk[r] = by_risk.get(r, 0) + 1
                by_region[reg] = by_region.get(reg, 0) + 1
            return {
                "total": len(docs),
                "by_risk": by_risk,
                "by_region": by_region,
                "storage": "memory",
            }


def _flatten_for_index(doc: dict) -> dict:
    """Flatten nested diagnosis for ES indexing."""
    diag = doc.get("diagnosis", {})
    flat = {**doc}
    flat["region"] = diag.get("region", "")
    flat["risk_level"] = diag.get("risk_level", "")
    flat["confidence"] = diag.get("confidence", 0)
    flat["summary"] = diag.get("summary", "")
    flat["findings_text"] = _build_findings_text(diag)
    return flat


def _build_findings_text(diag: dict) -> str:
    parts = []
    for f in diag.get("findings", []):
        parts.append(f.get("description", ""))
    for c in diag.get("areas_of_concern", []):
        parts.append(c.get("observation", ""))
    return " ".join(filter(None, parts))


def _build_search_text(doc: dict) -> str:
    diag = doc.get("diagnosis", {})
    parts = [
        doc.get("original_filename", ""),
        diag.get("region", ""),
        diag.get("summary", ""),
        diag.get("risk_level", ""),
        _build_findings_text(diag),
    ]
    return " ".join(filter(None, parts))

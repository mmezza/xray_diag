"""
Elasticsearch service with vector search (kNN) and in-memory fallback.

Architecture:
  - Each X-ray document stores a 1536-d dense_vector (text-embedding-3-small)
    generated from the diagnostic text (region, findings, summary, risk).
  - Search uses HYBRID mode: kNN (semantic, 60%) + BM25 (keyword, 40%).
  - /similar endpoint uses pure kNN to find the most analogous diagnoses.
  - When ES is not configured, falls back to in-memory cosine similarity.

Index mapping highlights:
  - embedding:      dense_vector (1536d, cosine) — kNN vector search
  - summary:        text, portuguese analyzer    — BM25 full-text
  - findings_text:  text, portuguese analyzer    — BM25 full-text
  - risk_level:     keyword                      — filtering / aggregations
  - region:         keyword                      — filtering / aggregations
"""

import logging
import math
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

INDEX_NAME    = "xray_diagnostics"
EMBEDDING_DIMS = 1536

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "id":                {"type": "keyword"},
            "original_filename": {"type": "keyword"},
            "stored_filename":   {"type": "keyword"},
            "media_type":        {"type": "keyword"},
            "upload_date":       {"type": "date"},
            "file_size_bytes":   {"type": "long"},
            "region":            {"type": "keyword"},
            "risk_level":        {"type": "keyword"},
            "confidence":        {"type": "float"},
            "summary":           {"type": "text", "analyzer": "portuguese"},
            "findings_text":     {"type": "text", "analyzer": "portuguese"},
            "diagnosis":         {"type": "object", "enabled": True},
            # ── vector field ──────────────────────────────────────────────
            "embedding": {
                "type":       "dense_vector",
                "dims":       EMBEDDING_DIMS,
                "index":      True,
                "similarity": "cosine",
            },
        }
    },
    "settings": {
        "number_of_shards":   1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "portuguese": {
                    "tokenizer": "standard",
                    "filter": ["lowercase", "portuguese_stop", "portuguese_stem"],
                }
            },
            "filter": {
                "portuguese_stop": {"type": "stop",    "stopwords": "_portuguese_"},
                "portuguese_stem": {"type": "stemmer", "language": "portuguese"},
            },
        },
    },
}

# exclude heavy embedding vector from _source by default
_SOURCE_EXCLUDE = {"excludes": ["embedding"]}


class ElasticsearchService:
    def __init__(self):
        self._es            = None
        self._memory_store: dict[str, dict] = {}   # id → doc (with embedding)
        self._es_available  = False

    # ── initialisation ───────────────────────────────────────────────────

    async def initialize(self):
        if not settings.elasticsearch_url:
            logger.info("Elasticsearch não configurado — modo memória com cosine similarity.")
            return
        try:
            from elasticsearch import AsyncElasticsearch
            kwargs: dict = {"hosts": [settings.elasticsearch_url]}
            if settings.elasticsearch_api_key:
                kwargs["api_key"] = settings.elasticsearch_api_key
            elif settings.elasticsearch_username and settings.elasticsearch_password:
                kwargs["basic_auth"] = (
                    settings.elasticsearch_username,
                    settings.elasticsearch_password,
                )
            self._es = AsyncElasticsearch(**kwargs)
            if not await self._es.ping():
                logger.warning("Elasticsearch não respondeu — modo memória.")
                self._es = None
                return
            await self._ensure_index()
            self._es_available = True
            logger.info("Elasticsearch conectado. Vector search (kNN) ativo.")
        except Exception as exc:
            logger.warning(f"Falha ao conectar ao Elasticsearch: {exc} — modo memória.")
            self._es = None

    async def _ensure_index(self):
        exists = await self._es.indices.exists(index=INDEX_NAME)
        if not exists:
            await self._es.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)
            logger.info(f"Índice '{INDEX_NAME}' criado com suporte a kNN.")
            return
        # Add embedding field to existing index if missing
        try:
            await self._es.indices.put_mapping(
                index=INDEX_NAME,
                body={"properties": {"embedding": INDEX_MAPPING["mappings"]["properties"]["embedding"]}},
            )
            logger.info("Campo 'embedding' adicionado ao índice existente.")
        except Exception:
            pass  # already exists

    # ── basic ops ────────────────────────────────────────────────────────

    async def ping(self) -> bool:
        if self._es:
            try:
                return await self._es.ping()
            except Exception:
                return False
        return False

    async def store_document(self, doc_id: str, doc: dict, embedding: Optional[list[float]] = None):
        """Store a document. Pass the embedding vector to enable vector search."""
        flat = _flatten_for_index(doc)
        if embedding:
            flat["embedding"] = embedding

        if self._es_available and self._es:
            await self._es.index(index=INDEX_NAME, id=doc_id, document=flat)
        else:
            # keep full doc + embedding in memory
            mem_doc = {**doc}
            if embedding:
                mem_doc["_embedding"] = embedding
            self._memory_store[doc_id] = mem_doc

    async def get_document(self, doc_id: str) -> Optional[dict]:
        if self._es_available and self._es:
            try:
                res = await self._es.get(index=INDEX_NAME, id=doc_id,
                                          source_excludes=["embedding"])
                return res["_source"]
            except Exception:
                return None
        doc = self._memory_store.get(doc_id)
        if doc:
            return {k: v for k, v in doc.items() if k != "_embedding"}
        return None

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
                    "sort":  [{"upload_date": {"order": "desc"}}],
                    "size":  size,
                    "_source": _SOURCE_EXCLUDE,
                },
            )
            return [hit["_source"] for hit in res["hits"]["hits"]]
        docs = list(self._memory_store.values())
        docs.sort(key=lambda d: d.get("upload_date", ""), reverse=True)
        return [{k: v for k, v in d.items() if k != "_embedding"} for d in docs[:size]]

    # ── vector search ────────────────────────────────────────────────────

    async def search(self, query: str, query_embedding: Optional[list[float]] = None,
                     size: int = 10) -> list[dict]:
        """
        Hybrid search: kNN (60%) + BM25 (40%) when ES + embedding available.
        Falls back to BM25-only (ES) or substring match (memory).
        """
        if self._es_available and self._es:
            if query_embedding:
                body = {
                    "knn": {
                        "field":         "embedding",
                        "query_vector":  query_embedding,
                        "k":             size,
                        "num_candidates": size * 10,
                        "boost":         0.6,
                    },
                    "query": {
                        "multi_match": {
                            "query":     query,
                            "fields":    ["summary^3", "findings_text^2", "region", "risk_level"],
                            "fuzziness": "AUTO",
                            "boost":     0.4,
                        }
                    },
                    "size":    size,
                    "_source": _SOURCE_EXCLUDE,
                }
            else:
                body = {
                    "query": {
                        "multi_match": {
                            "query":     query,
                            "fields":    ["summary^3", "findings_text^2", "region",
                                          "risk_level", "original_filename"],
                            "fuzziness": "AUTO",
                        }
                    },
                    "size":    size,
                    "_source": _SOURCE_EXCLUDE,
                    "highlight": {"fields": {"summary": {}, "findings_text": {}}},
                }

            res     = await self._es.search(index=INDEX_NAME, body=body)
            results = []
            for hit in res["hits"]["hits"]:
                doc               = hit["_source"]
                doc["_score"]     = round(hit["_score"], 4)
                doc["_search_mode"] = "hybrid" if query_embedding else "bm25"
                doc["_highlights"] = hit.get("highlight", {})
                results.append(doc)
            return results

        # ── in-memory fallback ──
        q = query.lower()
        scored = []
        for doc in self._memory_store.values():
            emb = doc.get("_embedding")
            if query_embedding and emb:
                score = _cosine(query_embedding, emb)
            else:
                score = 1.0 if q in _build_search_text(doc).lower() else 0.0
            if score > 0.0:
                clean = {k: v for k, v in doc.items() if k != "_embedding"}
                clean["_score"]       = round(score, 4)
                clean["_search_mode"] = "cosine-memory" if query_embedding else "text-memory"
                scored.append(clean)
        scored.sort(key=lambda d: d["_score"], reverse=True)
        return scored[:size]

    async def find_similar(self, doc_id: str, size: int = 5) -> list[dict]:
        """
        Return the `size` most similar cases to `doc_id` using kNN cosine search.
        Excludes the query document itself from results.
        """
        if self._es_available and self._es:
            # Fetch the embedding of the source document
            try:
                src = await self._es.get(index=INDEX_NAME, id=doc_id,
                                          source_includes=["embedding"])
                embedding = src["_source"].get("embedding")
            except Exception:
                return []

            if not embedding:
                return []

            body = {
                "knn": {
                    "field":          "embedding",
                    "query_vector":   embedding,
                    "k":              size + 1,
                    "num_candidates": (size + 1) * 10,
                    "filter": {
                        "bool": {"must_not": [{"term": {"id": doc_id}}]}
                    },
                },
                "size":    size,
                "_source": _SOURCE_EXCLUDE,
            }
            res = await self._es.search(index=INDEX_NAME, body=body)
            results = []
            for hit in res["hits"]["hits"]:
                if hit["_id"] == doc_id:
                    continue
                doc                    = hit["_source"]
                doc["_similarity"]     = round(hit["_score"], 4)
                doc["_search_mode"]    = "knn-cosine"
                results.append(doc)
            return results[:size]

        # ── in-memory cosine fallback ──
        source = self._memory_store.get(doc_id)
        if not source:
            return []
        src_emb = source.get("_embedding")
        if not src_emb:
            return []

        scored = []
        for did, doc in self._memory_store.items():
            if did == doc_id:
                continue
            emb = doc.get("_embedding")
            if not emb:
                continue
            sim = _cosine(src_emb, emb)
            clean                = {k: v for k, v in doc.items() if k != "_embedding"}
            clean["_similarity"] = round(sim, 4)
            clean["_search_mode"] = "cosine-memory"
            scored.append(clean)

        scored.sort(key=lambda d: d["_similarity"], reverse=True)
        return scored[:size]

    # ── aggregations ─────────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        if self._es_available and self._es:
            res  = await self._es.search(
                index=INDEX_NAME,
                body={
                    "size": 0,
                    "aggs": {
                        "by_risk":   {"terms": {"field": "risk_level"}},
                        "by_region": {"terms": {"field": "region"}},
                    },
                },
            )
            aggs = res.get("aggregations", {})
            return {
                "total":     res["hits"]["total"]["value"],
                "by_risk":   {b["key"]: b["doc_count"] for b in aggs.get("by_risk",   {}).get("buckets", [])},
                "by_region": {b["key"]: b["doc_count"] for b in aggs.get("by_region", {}).get("buckets", [])},
                "storage":   "elasticsearch",
                "vector_search": True,
            }
        docs = list(self._memory_store.values())
        by_risk:   dict[str, int] = {}
        by_region: dict[str, int] = {}
        has_vectors = 0
        for doc in docs:
            diag = doc.get("diagnosis", {})
            r    = diag.get("risk_level", "Desconhecido")
            reg  = diag.get("region",     "Desconhecido")
            by_risk[r]   = by_risk.get(r, 0)   + 1
            by_region[reg] = by_region.get(reg, 0) + 1
            if doc.get("_embedding"):
                has_vectors += 1
        return {
            "total":        len(docs),
            "by_risk":      by_risk,
            "by_region":    by_region,
            "storage":      "memory",
            "vector_search": has_vectors > 0,
            "vectors_indexed": has_vectors,
        }


# ── helpers ──────────────────────────────────────────────────────────────

def _flatten_for_index(doc: dict) -> dict:
    diag = doc.get("diagnosis", {})
    flat = {**doc}
    flat["region"]        = diag.get("region",     "")
    flat["risk_level"]    = diag.get("risk_level",  "")
    flat["confidence"]    = diag.get("confidence",   0)
    flat["summary"]       = diag.get("summary",     "")
    flat["findings_text"] = _build_findings_text(diag)
    return flat


def _build_findings_text(diag: dict) -> str:
    parts = []
    for f in diag.get("findings",          []):
        parts.append(f.get("description", ""))
    for c in diag.get("areas_of_concern", []):
        parts.append(c.get("observation",  ""))
    return " ".join(filter(None, parts))


def _build_search_text(doc: dict) -> str:
    diag  = doc.get("diagnosis", {})
    parts = [
        doc.get("original_filename", ""),
        diag.get("region",     ""),
        diag.get("summary",    ""),
        diag.get("risk_level", ""),
        _build_findings_text(diag),
    ]
    return " ".join(filter(None, parts))


def _cosine(a: list[float], b: list[float]) -> float:
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

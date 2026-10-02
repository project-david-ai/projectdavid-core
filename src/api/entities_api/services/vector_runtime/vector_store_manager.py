"""Core-owned Qdrant storage adapter; no embedding compute."""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from projectdavid_common import UtilsInterface
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant  # unified import
from qdrant_client.models import Filter

from .base_vector_store import (
    BaseVectorStore,
    StoreExistsError,
    StoreNotFoundError,
    VectorStoreError,
)

load_dotenv()
log = UtilsInterface.LoggingUtility()


class VectorStoreManager(BaseVectorStore):
    # ------------------------------------------------------------------ #
    # lifecycle helpers
    # ------------------------------------------------------------------ #
    def __init__(self, qdrant_url: Optional[str] = None):
        self.client = QdrantClient(
            url=qdrant_url or os.getenv("QDRANT_URL", "http://qdrant:6333")
        )

    @staticmethod
    def _generate_vector_id() -> str:
        return str(uuid.uuid4())

    # ------------------------------------------------------------------ #
    # collection management
    # ------------------------------------------------------------------ #
    def create_store(
        self,
        store_name: str,
        vector_size: int = 384,
        distance: str = "COSINE",
        vectors_config: Optional[Dict[str, qdrant.VectorParams]] = None,
    ) -> dict:
        """
        Create a Qdrant collection without replacing existing data.

        • If *vectors_config* is provided → use it verbatim (multi-vector schema).
        • Otherwise create a classic single-vector collection *without* naming the
          vector field – so upserts can omit ``vector_name``.
        """
        try:
            # ── pre-existence check ────────────────────────────────────────────
            if self.client.collection_exists(store_name):
                raise StoreExistsError(f"Collection '{store_name}' already exists")

            dist = distance.upper()
            if dist not in qdrant.Distance.__members__:
                raise ValueError(f"Invalid distance metric '{distance}'")

            # ── choose schema ──────────────────────────────────────────────────
            config: Union[qdrant.VectorParams, Dict[str, qdrant.VectorParams]]
            if vectors_config:  # caller supplied full mapping
                config = vectors_config  # e.g. {"text_vec": ..., "img_vec": ...}
            else:  # default = single unnamed vector
                config = qdrant.VectorParams(
                    size=vector_size,
                    distance=qdrant.Distance[dist],
                )

            # Create without replacing existing data.
            self.client.create_collection(
                collection_name=store_name,
                vectors_config=config,
            )

            log.info("Created Qdrant collection %s", store_name)
            return {"collection_name": store_name, "status": "created"}

        except (StoreExistsError, ValueError):
            raise
        except Exception as e:
            log.error("Create store failed: %s", e)
            raise VectorStoreError(f"Qdrant collection creation failed: {e}") from e

    def delete_store(self, store_name: str) -> dict:
        if not self.client.collection_exists(store_name):
            return {"name": store_name, "status": "absent"}
        try:
            self.client.delete_collection(collection_name=store_name)
            return {"name": store_name, "status": "deleted"}
        except Exception as e:
            log.error("Delete failed: %s", e)
            raise VectorStoreError(f"Store deletion failed: {e}") from e

    def get_store_info(self, store_name: str) -> dict:
        if not self.client.collection_exists(store_name):
            raise StoreNotFoundError(store_name)
        try:
            info = self.client.get_collection(collection_name=store_name)
            return {
                "name": store_name,
                "status": "active",
                "vectors_count": info.points_count,
                "configuration": info.config.params,
                "fields": (
                    list(info.config.params.vectors)
                    if isinstance(info.config.params.vectors, dict)
                    else []
                ),
            }
        except Exception as e:
            log.error("Store info failed: %s", e)
            raise VectorStoreError(f"Info retrieval failed: {e}") from e

    def add_to_store(
        self,
        store_name: str,
        texts: List[str],
        vectors: List[List[float]],
        metadata: List[dict],
        vector_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upsert vectors + payloads into *store_name*.

        If *vector_name* is omitted the manager:

        • auto-detects the single vector field for classic (unnamed) collections
        • auto-detects the sole key for named-vector collections with exactly one field
        • raises if multiple named fields exist.
        """

        # ─── input validation ───────────────────────────────────────────────
        if not vectors or len(texts) != len(vectors) or len(metadata) != len(vectors):
            raise ValueError(
                "texts, vectors and metadata must have equal non-zero lengths"
            )
        v_cfg = self.client.get_collection(store_name).config.params.vectors
        if isinstance(v_cfg, dict):
            if vector_name is None and len(v_cfg) == 1:
                vector_name = next(iter(v_cfg))
            if vector_name not in v_cfg:
                raise ValueError("Specify a valid named vector field")
            size = v_cfg[vector_name].size
        else:
            if vector_name is not None:
                raise ValueError("Collection uses unnamed vectors")
            size = v_cfg.size
        if any(len(v) != size for v in vectors):
            raise ValueError("Vector dimensions do not match collection")

        points = [
            qdrant.PointStruct(
                id=self._generate_vector_id(),
                vector={vector_name: vec} if vector_name is not None else vec,
                payload={"text": txt, **meta},
            )
            for txt, vec, meta in zip(texts, vectors, metadata)
        ]

        upsert_kwargs = {"collection_name": store_name, "points": points, "wait": True}
        try:
            self.client.upsert(**upsert_kwargs)
            return {
                "status": "success",
                "points_inserted": len(points),
                "point_ids": [p.id for p in points],
            }
        except Exception as exc:  # noqa: BLE001
            log.error("Add-to-store failed: %s", exc, exc_info=True)
            raise VectorStoreError(f"Insertion failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # search / query
    # ------------------------------------------------------------------ #
    @staticmethod
    def _dict_to_filter(filters: dict) -> Filter:
        """Validate Qdrant filters, including nested conditions and ranges."""
        return Filter.model_validate(filters)

    def query_store(
        self,
        store_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
        *,
        vector_field: Optional[str] = None,
        score_threshold: float = 0.0,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """
        Run a similarity search against *store_name*.

        • Handles both modern Qdrant (v1.10+ using query_points) and legacy clients (using search).
        • `vector_field` lets you target a non-default vector column.
        """

        limit = limit or top_k
        flt = self._dict_to_filter(filters) if filters else None

        try:
            kwargs = dict(
                collection_name=store_name,
                query_filter=flt,
                limit=limit,
                offset=offset,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )
            if hasattr(self.client, "query_points"):
                res = self.client.query_points(
                    query=query_vector, using=vector_field, **kwargs
                ).points
            else:
                res = self.client.search(
                    query_vector=(
                        (vector_field, query_vector) if vector_field else query_vector
                    ),
                    **kwargs,
                )
        except Exception as exc:
            log.error("Query failed: %s", exc)
            raise VectorStoreError(f"Query failed: {exc}") from exc

        return [
            {
                "id": p.id,
                "score": p.score,
                "text": (p.payload or {}).get("text"),
                "metadata": {k: v for k, v in (p.payload or {}).items() if k != "text"},
            }
            for p in res
        ]

    # ------------------------------------------------------------------ #
    # point / file deletion helpers
    # ------------------------------------------------------------------ #
    def delete_points(self, store_name: str, point_ids: List[str]) -> None:
        self.client.delete(
            collection_name=store_name,
            points_selector=qdrant.PointIdsList(points=point_ids),
            wait=True,
        )

    def delete_file_from_store(self, store_name: str, file_path: str) -> dict:
        try:
            cond = qdrant.FieldCondition(
                key="file_path", match=qdrant.MatchValue(value=file_path)
            )
            self.client.delete(
                collection_name=store_name,
                points_selector=qdrant.FilterSelector(
                    filter=qdrant.Filter(must=[cond])
                ),
                wait=True,
            )
            return {
                "deleted_file": file_path,
                "store_name": store_name,
                "status": "success",
            }
        except Exception as e:
            log.error("File deletion failed: %s", e)
            raise VectorStoreError(f"File deletion failed: {e}") from e

    # ------------------------------------------------------------------ #
    # misc helpers
    # ------------------------------------------------------------------ #
    def list_store_files(self, store_name: str) -> List[str]:
        """Return distinct `file_path` payload values present in the collection."""
        try:
            seen = set()
            scroll = self.client.scroll(
                collection_name=store_name,
                with_payload=["file_path"],
                limit=100,
            )
            while True:
                for pt in scroll[0]:
                    if fp := pt.payload.get("file_path"):
                        seen.add(fp)
                if scroll[1] is None:
                    break
                scroll = self.client.scroll(
                    collection_name=store_name,
                    with_payload=["file_path"],
                    limit=100,
                    offset=scroll[1],
                )
            return sorted(seen)
        except Exception as e:
            log.error("List store files failed: %s", e)
            raise VectorStoreError(f"List files failed: {e}") from e

    def get_point_by_id(self, store_name: str, point_id: str) -> Dict[str, Any]:
        try:
            res = self.client.retrieve(collection_name=store_name, ids=[point_id])
            pts = res.get("result") if isinstance(res, dict) else res
            if not pts:
                raise VectorStoreError(f"Point '{point_id}' not found")
            pt = pts[0]
            return {
                "id": pt.id,
                "payload": pt.payload if pt.payload is not None else {},
                "vector": pt.vector,
            }
        except Exception as e:
            log.error("Get point failed: %s", e)
            raise VectorStoreError(f"Fetch failed: {e}") from e

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def get_client(self) -> QdrantClient:
        return self.client

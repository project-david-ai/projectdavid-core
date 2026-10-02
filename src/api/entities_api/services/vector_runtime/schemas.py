"""Bounded data-plane contracts; vectors are computed by the caller."""

import json
from typing import Annotated, Any

from projectdavid_common import ValidationInterface
from projectdavid_common.schemas.vectors_schema import (
    VectorStoreAddRequest,
    VectorStoreSearchResult,
)
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    model_validator,
)

Vector = Annotated[list[FiniteFloat], Field(min_length=1, max_length=4096)]


class VectorUpsert(VectorStoreAddRequest):
    model_config = ConfigDict(extra="forbid")
    texts: list[Annotated[str, Field(max_length=65536)]] = Field(
        min_length=1, max_length=256
    )
    vectors: list[Vector] = Field(min_length=1, max_length=256)
    meta_data: list[dict[str, Any]] = Field(
        min_length=1,
        max_length=256,
        validation_alias=AliasChoices("metadata", "meta_data"),
    )

    vector_name: str | None = Field(default=None, max_length=128)
    file: ValidationInterface.VectorStoreFileCreate | None = None

    @property
    def metadata(self):
        return self.meta_data

    @model_validator(mode="after")
    def validate_batch(self):
        if len(self.texts) != len(self.vectors) or len(self.metadata) != len(
            self.vectors
        ):
            raise ValueError("texts, vectors and metadata lengths must match")
        if len({len(v) for v in self.vectors}) != 1:
            raise ValueError("Vector dimensions must be consistent")
        if (
            len(json.dumps(self.model_dump(), ensure_ascii=False).encode())
            > 8 * 1024 * 1024
        ):
            raise ValueError("Vector batch exceeds 8 MiB")
        return self


class VectorUpsertResult(BaseModel):
    status: str
    points_inserted: int
    file: ValidationInterface.VectorStoreFileRead | None = None


class VectorSearch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query_vector: Vector
    top_k: int = Field(default=5, ge=1, le=100)
    filters: dict[str, Any] | None = None
    vector_field: str | None = Field(default=None, max_length=128)
    score_threshold: FiniteFloat = 0.0
    offset: int = Field(default=0, ge=0, le=10000)

    @model_validator(mode="after")
    def bound_filters(self):
        if len(json.dumps(self.filters).encode()) > 65536:
            raise ValueError("Search filters exceed 64 KiB")
        return self


class VectorHit(VectorStoreSearchResult):
    id: str | int
    vector_id: str | int
    score: float
    text: str | None = None
    metadata: dict[str, Any]
    meta_data: dict[str, Any]

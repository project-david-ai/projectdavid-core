from abc import ABC, abstractmethod
from typing import List, Optional

from qdrant_client import QdrantClient


class VectorStoreError(Exception):
    """Base exception for all vector store operations."""

    pass


class StoreExistsError(VectorStoreError):
    """Raised when attempting to create a store that already exists."""

    pass


class StoreNotFoundError(VectorStoreError):
    """Raised when a store is not found."""

    pass


class BaseVectorStore(ABC):
    @abstractmethod
    def create_store(self, store_name: str, vector_size: int, distance: str) -> dict:
        """Create a new vector store."""
        pass

    @abstractmethod
    def add_to_store(
        self,
        store_name: str,
        texts: List[str],
        vectors: List[List[float]],
        metadata: List[dict],
    ) -> dict:
        """Add entries to the vector store."""
        pass

    @abstractmethod
    def query_store(
        self,
        store_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> List[dict]:
        """Query the vector store with optional filters."""
        pass

    @abstractmethod
    def delete_store(self, store_name: str) -> dict:
        """Delete the specified vector store."""
        pass

    @abstractmethod
    def get_store_info(self, store_name: str) -> dict:
        """Retrieve metadata for the specified vector store."""
        pass

    @abstractmethod
    def list_store_files(self, store_name: str) -> List[str]:
        """List unique source files in the vector store."""
        pass

    @abstractmethod
    def delete_file_from_store(self, store_name: str, file_path: str) -> dict:
        """Delete all vectors associated with a specific file."""
        pass

    @abstractmethod
    def get_point_by_id(self, store_name: str, point_id: str) -> dict:
        """Retrieve a specific point by its ID."""
        pass

    @abstractmethod
    def get_client(self) -> QdrantClient:
        """Return the underlying Qdrant client."""
        pass

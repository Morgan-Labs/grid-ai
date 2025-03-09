"""Abstract base class for embedding services."""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingService(ABC):
    """Abstract base class for embedding services."""

    @abstractmethod
    async def get_embeddings(self, texts: List[str], parent_run_id: str = None) -> List[List[float]]:
        """Get the embeddings for the given text."""
        pass

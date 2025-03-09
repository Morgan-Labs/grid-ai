"""Abstract base class for language model completion services."""

from abc import ABC, abstractmethod
from typing import Any


class CompletionService(ABC):
    """Abstract base class for language model completion services."""

    @abstractmethod
    async def generate_completion(
        self, prompt: str, response_model: Any, parent_run_id: str = None
    ) -> Any:
        """Generate a completion from the language model."""
        pass

    @abstractmethod
    async def decompose_query(self, query: str, parent_run_id: str = None) -> dict[str, Any]:
        """Decompose the query into smaller sub-queries."""
        pass

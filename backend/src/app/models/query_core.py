"""Query model."""

from typing import Any, Dict, List, Literal, Optional, TypeAlias, Union

from pydantic import BaseModel, Field


class EntitySource(BaseModel):
    """Entity source model."""

    type: Literal["column", "global"]
    id: str


class ResolvedEntity(BaseModel):
    """Resolved entity model."""

    original: Union[str, List[str]]
    resolved: Union[str, List[str]]
    source: EntitySource
    entityType: str


class TransformationDict(BaseModel):
    """Transformation dictionary model."""

    original: Union[str, List[str]]
    resolved: Union[str, List[str]]


class Rule(BaseModel):
    """Rule model."""

    type: Literal["must_return", "may_return", "max_length", "resolve_entity"]
    options: Optional[List[str]] = None
    length: Optional[int] = None


class Chunk(BaseModel):
    """Chunk model."""

    id: str
    content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None


class Answer(BaseModel):
    """Answer model."""

    id: str
    document_id: str
    prompt_id: str
    answer: Optional[Union[int, str, bool, List[int], List[str]]]
    chunks: List[Chunk]
    type: str


QueryType: TypeAlias = Literal[
    "decomposition", "hybrid", "simple_vector", "inference"
]
FormatType: TypeAlias = Literal[
    "str", "int", "bool", "str_array", "int_array", "entity"
]


class QueryResult(BaseModel):
    """The result of a query."""

    answer: Any
    chunks: Optional[List[Chunk]] = []
    resolved_entities: Optional[List[Dict[str, Any]]] = []
    queued: bool = False
    queued_query: Optional[Any] = None

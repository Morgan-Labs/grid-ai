"""Query schemas for API requests and responses."""

from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, validator

from app.models.query_core import Chunk, FormatType, Rule


class ResolvedEntitySchema(BaseModel):
    """Schema for resolved entity transformations."""

    original: Union[str, List[str]]
    resolved: Union[str, List[str]]
    source: dict[str, str]
    entityType: str


class QueryPromptSchema(BaseModel):
    """Schema for the prompt part of the query request."""

    id: str
    entity_type: str
    query: str
    type: FormatType
    rules: list[Rule] = []


class QueryRequestSchema(BaseModel):
    """Query request schema."""

    document_id: str
    prompt: QueryPromptSchema

    model_config = ConfigDict(extra="allow")


class VectorResponseSchema(BaseModel):
    """Vector response schema."""

    message: str
    chunks: List[Chunk]
    keywords: Optional[List[str]] = None


class QueryResult(BaseModel):
    """Query result schema."""

    answer: Any
    chunks: List[Chunk]
    resolved_entities: Optional[List[ResolvedEntitySchema]] = None


class QueryResponseSchema(BaseModel):
    """Query response schema."""

    id: str
    document_id: str
    prompt_id: str
    answer: Optional[Any] = None
    chunks: List[Chunk]
    type: str
    resolved_entities: Optional[List[ResolvedEntitySchema]] = None


class QueryAnswer(BaseModel):
    """Query answer model."""

    id: str
    document_id: str
    prompt_id: str
    answer: Optional[Union[int, str, bool, List[int], List[str]]]
    type: str
    
    # Add a validator to ensure answer matches the specified type
    @validator('answer')
    def validate_answer_type(cls, v, values):
        """Validate that the answer matches the specified type."""
        if v is None:
            return v
            
        prompt_type = values.get('type', 'str')
        
        # Handle dictionary inputs (most common error case)
        if isinstance(v, dict) and 'answer' in v:
            # If we get {'answer': value} format, extract the value
            nested_value = v.get('answer')
            # Recursively validate with the extracted value
            return cls.validate_answer_type(nested_value, values)
            
        # Ensure type conversion based on prompt_type
        try:
            if prompt_type == 'int' and not isinstance(v, int):
                try:
                    # Try to convert to int
                    return int(v)
                except (ValueError, TypeError):
                    return 0  # Default integer fallback
                
            elif prompt_type == 'bool' and not isinstance(v, bool):
                # Handle common string representations of booleans
                if isinstance(v, str):
                    if v.lower() in ('true', 'yes', '1'):
                        return True
                    elif v.lower() in ('false', 'no', '0'):
                        return False
                return False  # Default boolean fallback
                
            elif prompt_type == 'int_array' and not isinstance(v, list):
                # If it's a single int, wrap it in a list
                if isinstance(v, int):
                    return [v]
                return [0]  # Default integer array fallback
                
            elif prompt_type == 'str_array' and not isinstance(v, list):
                # If it's a single string, wrap it in a list
                if isinstance(v, str):
                    return [v]
                return ["Error: Failed to process"]  # Default string array fallback
                
            elif prompt_type == 'str' and not isinstance(v, str):
                return str(v) if v is not None else "Error: No response"
            
            # If it's already the right type, return as is
            return v
        except Exception as e:
            # Provide type-appropriate fallback values on error
            if prompt_type == 'int':
                return 0
            elif prompt_type == 'bool':
                return False
            elif prompt_type == 'int_array':
                return [0]
            elif prompt_type == 'str_array':
                return ["Error: Failed to process"]
            else:
                return "Error: Failed to process"


class QueryAnswerResponse(BaseModel):
    """Query answer response model."""

    answer: QueryAnswer
    chunks: List[Chunk]
    resolved_entities: Optional[List[ResolvedEntitySchema]] = None


# Type for search responses (used in service layer)
SearchResponse = Union[dict[str, List[Chunk]], VectorResponseSchema]

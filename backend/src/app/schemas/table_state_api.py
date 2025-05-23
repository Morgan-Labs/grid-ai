"""API schemas for table state operations."""

from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


class TableStateBase(BaseModel):
    name: str = Field(..., description="The name of the table state.")
    user_id: Optional[str] = Field(None, description="The ID of the user who owns this table state.")
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="The data of the table state, containing various table components."
    )


class TableStateCreate(TableStateBase):
    id: str = Field(..., description="The unique ID for this table state.")


class TableStateUpdate(BaseModel):
    name: Optional[str] = Field(None, description="The new name of the table state.")
    data: Optional[Dict[str, Any]] = Field(
        None,
        description="The new data of the table state."
    )


class TableStateResponse(TableStateBase):
    id: str = Field(..., description="The unique ID of this table state.")
    created_at: datetime = Field(..., description="The timestamp when the table state was created.")
    updated_at: datetime = Field(..., description="The timestamp when the table state was last updated.")

    class Config:
        from_attributes = True # Replace ormar_mode for Pydantic v2


class TableStateListResponse(BaseModel):
    items: List[TableStateResponse] = Field(..., description="A list of table states.")

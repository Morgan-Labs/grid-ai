"""API schemas for table state operations."""

from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import BaseModel


class TableStateCreate(BaseModel):
    """Schema for creating a new table state."""
    
    id: str
    name: str
    data: Dict[str, Any]


class TableStateUpdate(BaseModel):
    """Schema for updating an existing table state."""
    
    name: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class TableStateResponse(BaseModel):
    """Schema for table state response."""
    
    id: str
    name: str
    data: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class TableStateListItem(BaseModel):
    """Schema for table state list item (without full data)."""
    id: str
    name: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TableStateListResponse(BaseModel):
    """Schema for listing table states."""
    
    items: List[TableStateListItem]

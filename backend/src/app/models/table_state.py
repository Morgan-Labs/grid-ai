"""Table state model for storing and retrieving table data."""

from datetime import datetime
from typing import Dict, Optional, Any

from pydantic import BaseModel, Field


class TableState(BaseModel):
    """Model for storing table state data."""
    
    id: str
    name: str
    user_id: Optional[str] = None  # For future multi-user support
    data: Dict[str, Any]  # The complete table state as JSON
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Configuration for the TableState model."""
        
        from_attributes = True  # Updated from orm_mode for Pydantic V2

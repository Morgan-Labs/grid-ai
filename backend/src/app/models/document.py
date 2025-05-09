"""Document model."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Document model."""

    id: str = Field(description="Document ID.")
    name: str = Field(description="Document name.")
    author: str = Field(description="Document author.")
    tag: str = Field(description="Document tag.")
    page_count: int = Field(description="Document page count.")
    status: str = Field(default="completed", description="Document processing status: processing, completed, or failed.")

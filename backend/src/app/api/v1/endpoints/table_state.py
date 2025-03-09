"""API endpoints for table state operations."""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from fastapi.responses import JSONResponse

from app.core.auth import jwt_auth
from app.core.config import get_settings
from app.models.table_state import TableState
from app.schemas.table_state_api import (
    TableStateCreate,
    TableStateResponse,
    TableStateUpdate,
    TableStateListResponse,
)
from app.services.table_state_service import TableStateService

# Set up logging
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Create router
router = APIRouter()


@router.post(
    "/",
    response_model=TableStateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new table state",
    description="Create a new table state with the provided data.",
)
async def create_table_state(
    table_state_create: TableStateCreate,
    _: dict = Depends(jwt_auth),
) -> TableStateResponse:
    """Create a new table state."""
    # Check if a table state with the same ID already exists
    existing_table_state = TableStateService.get_table_state(table_state_create.id)
    if existing_table_state:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Table state with ID {table_state_create.id} already exists",
        )
    
    # Create a new table state
    table_state = TableState(
        id=table_state_create.id,
        name=table_state_create.name,
        data=table_state_create.data,
    )
    
    # Save the table state
    saved_table_state = TableStateService.save_table_state(table_state)
    
    return TableStateResponse(
        id=saved_table_state.id,
        name=saved_table_state.name,
        data=saved_table_state.data,
        created_at=saved_table_state.created_at,
        updated_at=saved_table_state.updated_at,
    )


@router.get(
    "/",
    response_model=TableStateListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all table states",
    description="List all table states.",
)
async def list_table_states(
    _: dict = Depends(jwt_auth),
) -> TableStateListResponse:
    """List all table states."""
    table_states = TableStateService.list_table_states()
    
    return TableStateListResponse(
        items=[
            TableStateResponse(
                id=table_state.id,
                name=table_state.name,
                data=table_state.data,
                created_at=table_state.created_at,
                updated_at=table_state.updated_at,
            )
            for table_state in table_states
        ]
    )


@router.get(
    "/{table_id}",
    response_model=TableStateResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a table state by ID",
    description="Get a table state by ID.",
)
async def get_table_state(
    table_id: str = Path(..., description="The ID of the table state to get"),
    _: dict = Depends(jwt_auth),
) -> TableStateResponse:
    """Get a table state by ID."""
    table_state = TableStateService.get_table_state(table_id)
    
    if not table_state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Table state with ID {table_id} not found",
        )
    
    return TableStateResponse(
        id=table_state.id,
        name=table_state.name,
        data=table_state.data,
        created_at=table_state.created_at,
        updated_at=table_state.updated_at,
    )


@router.put(
    "/{table_id}",
    response_model=TableStateResponse,
    status_code=status.HTTP_200_OK,
    summary="Update a table state by ID",
    description="Update a table state by ID.",
)
async def update_table_state(
    table_state_update: TableStateUpdate,
    table_id: str = Path(..., description="The ID of the table state to update"),
    _: dict = Depends(jwt_auth),
) -> TableStateResponse:
    """Update a table state by ID."""
    # Get the existing table state
    existing_table_state = TableStateService.get_table_state(table_id)
    
    if not existing_table_state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Table state with ID {table_id} not found",
        )
    
    # Update the table state
    if table_state_update.name is not None:
        existing_table_state.name = table_state_update.name
    
    if table_state_update.data is not None:
        existing_table_state.data = table_state_update.data
    
    # Save the updated table state
    updated_table_state = TableStateService.save_table_state(existing_table_state)
    
    return TableStateResponse(
        id=updated_table_state.id,
        name=updated_table_state.name,
        data=updated_table_state.data,
        created_at=updated_table_state.created_at,
        updated_at=updated_table_state.updated_at,
    )


@router.delete(
    "/{table_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a table state by ID",
    description="Delete a table state by ID.",
)
async def delete_table_state(
    table_id: str = Path(..., description="The ID of the table state to delete"),
    _: dict = Depends(jwt_auth),
) -> None:
    """Delete a table state by ID."""
    # Delete the table state
    deleted = TableStateService.delete_table_state(table_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Table state with ID {table_id} not found",
        )

"""API for the AI Grid."""

from fastapi import APIRouter

from app.api.v1.endpoints import auth, document, graph, query, table_state

api_router = APIRouter()
api_router.include_router(
    auth.router, prefix="/auth", tags=["auth"]
)
api_router.include_router(
    document.router, prefix="/document", tags=["document"]
)
api_router.include_router(graph.router, prefix="/graph", tags=["graph"])
api_router.include_router(query.router, prefix="/query", tags=["query"])
api_router.include_router(
    table_state.router, prefix="/table-state", tags=["table-state"]
)

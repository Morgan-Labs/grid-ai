"""Unstructured loader service with local processing."""

import asyncio
import logging
from typing import List

from langchain.schema import Document as LangchainDocument

from app.core.config import Settings
from app.services.loaders.base import LoaderService

logger = logging.getLogger(__name__)

# Try to import UnstructuredPDFLoader, but don't fail if it's not available
try:
    from langchain_community.document_loaders import UnstructuredPDFLoader
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logger.warning("unstructured package not found, please install it with `pip install unstructured`")


class UnstructuredLoader(LoaderService):
    """Unstructured loader service with local processing."""

    def __init__(self, settings: Settings):
        """Initialize the UnstructuredLoader."""
        if not UNSTRUCTURED_AVAILABLE:
            raise ImportError(
                "The 'unstructured' package is not installed. "
                "Please install it with `pip install unstructured`"
            )
        self.settings = settings

    async def load(self, file_path: str) -> List[LangchainDocument]:
        """Load document from file path using local Unstructured processing."""
        if not UNSTRUCTURED_AVAILABLE:
            raise ImportError(
                "The 'unstructured' package is not installed. "
                "Please install it with `pip install unstructured`"
            )
        
        try:
            logger.info(f"Attempting to load document with UnstructuredPDFLoader: {file_path}")
            loop = asyncio.get_event_loop()
            
            # Try different configurations to see what works
            try:
                # First try with default parameters
                logger.info("Trying UnstructuredPDFLoader with default parameters")
                unstructured_loader = UnstructuredPDFLoader(file_path)
                unstructured_documents = await loop.run_in_executor(None, unstructured_loader.load)
                
                if unstructured_documents:
                    logger.info(f"Successfully loaded document with default UnstructuredPDFLoader: {len(unstructured_documents)} elements")
                    return unstructured_documents
                else:
                    logger.warning("Default UnstructuredPDFLoader returned empty content")
            except Exception as e1:
                logger.error(f"Error using default UnstructuredPDFLoader: {str(e1)}")
                logger.error(f"Error type: {type(e1).__name__}")
                logger.error(f"Error details: {repr(e1)}")
            
            try:
                # Try with elements mode and fast strategy
                logger.info("Trying UnstructuredPDFLoader with elements mode and fast strategy")
                unstructured_loader = UnstructuredPDFLoader(
                    file_path,
                    mode="elements",
                    strategy="fast"
                )
                
                # Execute in thread pool to avoid blocking
                unstructured_documents = await loop.run_in_executor(None, unstructured_loader.load)
                
                if unstructured_documents:
                    logger.info(f"Successfully loaded document with elements/fast UnstructuredPDFLoader: {len(unstructured_documents)} elements")
                    return unstructured_documents
                else:
                    logger.warning("Elements/fast UnstructuredPDFLoader returned empty content")
            except Exception as e2:
                logger.error(f"Error using elements/fast UnstructuredPDFLoader: {str(e2)}")
                logger.error(f"Error type: {type(e2).__name__}")
                logger.error(f"Error details: {repr(e2)}")
            
            try:
                # Try with paged mode
                logger.info("Trying UnstructuredPDFLoader with paged mode")
                unstructured_loader = UnstructuredPDFLoader(
                    file_path,
                    mode="paged"
                )
                
                # Execute in thread pool to avoid blocking
                unstructured_documents = await loop.run_in_executor(None, unstructured_loader.load)
                
                if unstructured_documents:
                    logger.info(f"Successfully loaded document with paged UnstructuredPDFLoader: {len(unstructured_documents)} elements")
                    return unstructured_documents
                else:
                    logger.warning("Paged UnstructuredPDFLoader returned empty content")
            except Exception as e3:
                logger.error(f"Error using paged UnstructuredPDFLoader: {str(e3)}")
                logger.error(f"Error type: {type(e3).__name__}")
                logger.error(f"Error details: {repr(e3)}")
            
            # If all attempts failed, return empty list
            logger.error("All UnstructuredPDFLoader configurations failed")
            return []
            
        except Exception as e:
            logger.error(f"Error in UnstructuredLoader.load: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {repr(e)}")
            return []

"""Loader factory."""

import logging
from typing import Optional

from app.core.config import Settings
from app.services.loaders.base import LoaderService
from app.services.loaders.pypdf_service import PDFLoader
from app.services.loaders.pymupdf_service import PyMuPDFLoaderService
from app.services.loaders.enhanced_pdf_service import EnhancedPDFLoader
from app.services.loaders.textract_service import TextractLoader
from app.services.loaders.gpt4o_pdf_service import GPT4OPDFLoader
from app.services.loaders.simple_pdf_service import SimplePDFLoader

logger = logging.getLogger(__name__)

# Attempt to import UnstructuredLoader, but don't raise an error if it fails
try:
    from app.services.loaders.unstructured_service import UnstructuredLoader

    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logger.warning(
        "UnstructuredLoader is not available. Install the 'unstructured' extra to use it."
    )


class LoaderFactory:
    """The factory for the loader services."""

    @staticmethod
    def create_loader(settings: Settings) -> Optional[LoaderService]:
        """Create a loader service."""
        loader_type = settings.loader
        logger.info(f"Creating loader of type: {loader_type}")

        if loader_type == "unstructured":
            if not UNSTRUCTURED_AVAILABLE:
                logger.warning(
                    "The 'unstructured' package is not installed. "
                    "Please install it with `pip install unstructured`"
                )
                return None
            logger.info("Using UnstructuredLoader with local processing")
            return UnstructuredLoader(settings=settings)
        elif loader_type == "pypdf":
            logger.info("Using PyPDFLoader")
            return PDFLoader()
        elif loader_type == "pymupdf":
            logger.info("Using PyMuPDFLoader with enhanced OCR capabilities")
            return PyMuPDFLoaderService(settings=settings)
        elif loader_type == "gpt4o_pdf":
            logger.info("Using GPT-4o enhanced PDF loader")
            if not settings.openai_api_key:
                logger.warning("OpenAI API key not provided for GPT-4o PDF loader")
            return GPT4OPDFLoader(settings=settings)
        elif loader_type == "enhanced_pdf":
            logger.info("Using EnhancedPDFLoader")
            return EnhancedPDFLoader()
        elif loader_type == "textract":
            logger.info("Using Amazon Textract Loader")
            if not settings.aws_access_key_id or not settings.aws_secret_access_key:
                logger.warning("AWS credentials not provided for Textract loader")
            return TextractLoader(settings=settings)
        elif loader_type == "simple_pdf":
            logger.info("Using SimplePDFLoader")
            return SimplePDFLoader()
        else:
            logger.warning(f"No loader found for type: {loader_type}")
            return None

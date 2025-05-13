"""Caching service for document embeddings."""

import hashlib
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache for document embeddings to avoid redundant API calls."""
    
    def __init__(self, max_size: int = 10000):
        """Initialize the embedding cache.
        
        Parameters
        ----------
        max_size : int
            Maximum number of embeddings to store in cache
        """
        self.cache: Dict[str, List[float]] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache if it exists.
        
        Parameters
        ----------
        text : str
            The text to look up in the cache
            
        Returns
        -------
        Optional[List[float]]
            The embedding if found in cache, None otherwise
        """
        text_hash = self._hash_text(text)
        result = self.cache.get(text_hash)
        
        if result is not None:
            self.hits += 1
        else:
            self.misses += 1
            
        return result
    
    def set(self, text: str, embedding: List[float]) -> None:
        """Store embedding in cache.
        
        Parameters
        ----------
        text : str
            The text to cache
        embedding : List[float]
            The embedding to store
        """
        # Simple LRU-like eviction if cache is full
        if len(self.cache) >= self.max_size:
            # Remove a random item (first item in dict)
            self.cache.pop(next(iter(self.cache)))
            logger.debug(f"Cache full, evicted one item (size: {len(self.cache)})")
            
        text_hash = self._hash_text(text)
        self.cache[text_hash] = embedding
    
    def _hash_text(self, text: str) -> str:
        """Create a hash of the text for cache key.
        
        Parameters
        ----------
        text : str
            The text to hash
            
        Returns
        -------
        str
            MD5 hash of the text
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics.
        
        Returns
        -------
        Dict[str, int]
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        logger.info("Embedding cache cleared")
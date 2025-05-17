import { checkDocumentStatus } from '@config/api';

interface CachedStatus {
  status: string;
  timestamp: number;
  checkCount: number;
}

/**
 * Document status cache with TTL and exponential backoff
 * This helps reduce API calls and provides a fallback when API is unavailable
 */
class DocumentStatusCache {
  private cache = new Map<string, CachedStatus>();
  private readonly BASE_CACHE_TTL = 60000; // 1 minute base cache TTL
  private readonly MAX_CACHE_TTL = 180000; // 3 minutes maximum cache TTL
  private readonly MAX_CHECK_COUNT = 10; // Maximum number of checks before using max TTL
  private inProgressChecks = new Set<string>(); // Track in-progress status checks
  
  /**
   * Calculate cache TTL based on check count with exponential backoff
   * @param checkCount Number of times the status has been checked
   * @returns TTL in milliseconds
   */
  private calculateCacheTTL(checkCount: number): number {
    // For processing documents, use exponential backoff
    if (checkCount < this.MAX_CHECK_COUNT) {
      // 2^checkCount * BASE_CACHE_TTL (e.g., 1min, 2min, 4min, 8min, etc.)
      return Math.min(
        this.BASE_CACHE_TTL * Math.pow(2, checkCount),
        this.MAX_CACHE_TTL
      );
    }
    
    // After MAX_CHECK_COUNT checks, use the maximum TTL
    return this.MAX_CACHE_TTL;
  }
  
  /**
   * Get document status, either from cache or from API with exponential backoff
   * @param documentId Document ID
   * @returns Document status (completed, processing, or failed)
   */
  async getStatus(documentId: string): Promise<string> {
    const now = Date.now();
    const cached = this.cache.get(documentId);
    
    // If there's an in-progress check for this document, use cached value or default to unknown
    if (this.inProgressChecks.has(documentId)) {
      return cached?.status || 'unknown';  // Use 'unknown' instead of 'completed'
    }
    
    // Return cached value if still valid based on exponential backoff
    if (cached) {
      const ttl = this.calculateCacheTTL(cached.checkCount);
      
      // If document is completed or failed, use longer cache time
      if (cached.status !== 'processing') {
        // Completed/failed documents use max TTL
        if (now - cached.timestamp < this.MAX_CACHE_TTL) {
          return cached.status;
        }
      } else {
        // Processing documents use exponential backoff
        if (now - cached.timestamp < ttl) {
          return cached.status;
        }
      }
    }
    
    // Mark this document as being checked to prevent parallel checks
    this.inProgressChecks.add(documentId);
    
    try {
      // Add a small random delay to avoid thundering herd problem
      const jitter = Math.floor(Math.random() * 500);
      await new Promise(resolve => setTimeout(resolve, jitter));
      
      // Fetch from API
      const result = await checkDocumentStatus(documentId);
      
      // Update cache with incremented check count
      this.cache.set(documentId, {
        status: result.status,
        timestamp: now,
        checkCount: (cached?.checkCount || 0) + 1
      });
      
      return result.status;
    } catch (error) {
      console.error(`Error fetching document status: ${error}`);
      
      // If we have a stale cache entry, use it as fallback but don't update timestamp
      if (cached) {
        return cached.status;
      }
      
      // Default to unknown on error, not completed
      return 'unknown';
    } finally {
      // Remove from in-progress checks
      this.inProgressChecks.delete(documentId);
    }
  }
  
  /**
   * Update cache with a known status
   * @param documentId Document ID
   * @param status Document status
   */
  updateCache(documentId: string, status: string): void {
    const cached = this.cache.get(documentId);
    
    this.cache.set(documentId, {
      status,
      timestamp: Date.now(),
      checkCount: cached?.checkCount || 0
    });
  }
  
  /**
   * Clear cache for a specific document
   * @param documentId Document ID
   */
  clearCache(documentId: string): void {
    this.cache.delete(documentId);
    this.inProgressChecks.delete(documentId);
  }
  
  /**
   * Clear entire cache
   */
  clearAllCache(): void {
    this.cache.clear();
    this.inProgressChecks.clear();
  }
  
  /**
   * Get current check count for a document
   * Useful for debugging and testing
   * @param documentId Document ID
   * @returns Check count or 0 if not in cache
   */
  getCheckCount(documentId: string): number {
    return this.cache.get(documentId)?.checkCount || 0;
  }
}

// Export singleton instance
export const documentStatusCache = new DocumentStatusCache();
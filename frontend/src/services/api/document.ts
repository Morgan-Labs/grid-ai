import { API_ENDPOINTS, getAuthHeaders } from '../../config/api';

// Interface for Document status response
export interface DocumentStatusResponse {
  status: 'processing' | 'completed' | 'failed';
}

// Interface for Document metadata
export interface DocumentResponse {
  id: string;
  name: string;
  author?: string;
  tag?: string;
  page_count?: number;
  status: 'processing' | 'completed' | 'failed';
}

// Cache for document statuses to reduce API calls
const documentStatusCache: Record<string, { status: 'processing' | 'completed' | 'failed', timestamp: number }> = {};

// Cache time-to-live in milliseconds (5 minutes)
const CACHE_TTL = 5 * 60 * 1000;

/**
 * Get the status of a document by ID
 * @param documentId The document ID
 * @param forceRefresh Force a fresh API check regardless of cache
 * @returns Promise with document status
 */
export const getDocumentStatus = async (
  documentId: string, 
  forceRefresh = false
): Promise<DocumentStatusResponse> => {
  // Check cache first, unless forceRefresh is true
  if (!forceRefresh && documentStatusCache[documentId]) {
    const cachedData = documentStatusCache[documentId];
    const now = Date.now();
    
    // Only use cache if it's not expired and the status isn't "processing"
    // We always want fresh data for processing documents
    if (cachedData.status !== 'processing' && 
        now - cachedData.timestamp < CACHE_TTL) {
      return { status: cachedData.status };
    }
  }
  
  try {
    const response = await fetch(API_ENDPOINTS.DOCUMENT_STATUS(documentId), {
      method: 'GET',
      headers: getAuthHeaders(),
      credentials: 'include',
      mode: 'cors'
    });
    
    if (!response.ok) {
      if (response.status === 404) {
        // If the document doesn't exist, consider it completed by default
        const status = 'completed' as const;
        documentStatusCache[documentId] = { status, timestamp: Date.now() };
        return { status };
      }
      throw new Error(`Failed to get document status: ${response.statusText}`);
    }
    
    const result = await response.json();
    
    // Ensure status is always one of the valid values
    let status: 'processing' | 'completed' | 'failed';
    if (result.status === 'processing' || result.status === 'completed' || result.status === 'failed') {
      status = result.status;
    } else {
      // Default to completed for any other status
      // Only log if the status is a non-empty value (avoid log spam)
      if (result.status && result.status !== 'unknown') {
        console.warn(`Invalid status returned for document ${documentId}: ${result.status}, using 'completed' as fallback`);
      }
      status = 'completed';
    }
    
    // Update cache
    documentStatusCache[documentId] = { status, timestamp: Date.now() };
    
    return { status };
  } catch (error) {
    console.error('Error getting document status:', error);
    // Return a default in case of errors
    const status = 'completed' as const;
    documentStatusCache[documentId] = { status, timestamp: Date.now() };
    return { status };
  }
};

/**
 * Check if a document exists by ID and retrieve its details
 * @param documentId The document ID
 * @returns Promise with document metadata or null if not found
 */
export const getDocumentDetails = async (documentId: string): Promise<DocumentResponse | null> => {
  try {
    // We can create a specific endpoint for this in the backend later
    // For now, we'll use the document status endpoint
    const response = await fetch(API_ENDPOINTS.DOCUMENT_STATUS(documentId), {
      method: 'GET',
      headers: getAuthHeaders(),
      credentials: 'include',
      mode: 'cors'
    });
    
    if (!response.ok) {
      if (response.status === 404) {
        return null;
      }
      throw new Error(`Failed to get document details: ${response.statusText}`);
    }
    
    const statusResponse = await response.json();
    
    // Create a minimal document response with the status
    return {
      id: documentId,
      name: `document-${documentId}`, // Placeholder
      status: statusResponse.status
    };
  } catch (error) {
    console.error('Error getting document details:', error);
    return null;
  }
}; 
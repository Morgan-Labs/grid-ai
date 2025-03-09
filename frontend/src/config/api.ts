/**
 * API configuration for the frontend
 */
import { z } from 'zod';
// Get the API URL from environment variables or use a default
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
// Log the API URL for debugging
// console.log('API URL:', API_URL);
// API error class
export class ApiError extends Error {
  status: number;
  
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}
// API schemas
export const chunkSchema = z.object({
  id: z.string(),
  text: z.string(),
  content: z.string(),
  page: z.number(),
  metadata: z.record(z.any()).optional(),
});
export const documentSchema = z.object({
  id: z.string(),
  name: z.string(),
  page_count: z.number().optional(),
  author: z.string().optional(),
  tag: z.string().optional(),
  chunks: z.array(chunkSchema).optional(),
});
export const answerSchema = z.union([
  z.string(),
  z.number(),
  z.boolean(),
  z.array(z.string()),
  z.array(z.number()),
  z.object({
    text: z.string(),
    chunks: z.array(chunkSchema).optional(),
    resolvedEntities: z.array(z.any()).optional(),
  })
]);

// API endpoints
export const API_ENDPOINTS = {
  // Base API URL
  BASE_URL: API_URL,
  
  // API version path
  API_V1: `${API_URL}/api/v1`,
  
  // Document endpoints
  DOCUMENTS: `${API_URL}/api/v1/document`,
  DOCUMENT_UPLOAD: `${API_URL}/api/v1/document`,
  BATCH_DOCUMENT_UPLOAD: `${API_URL}/api/v1/document/batch`,
  DOCUMENT_PROCESS: `${API_URL}/api/v1/document/process`,
  DOCUMENT_PREVIEW: (id: string) => `${API_URL}/api/v1/document/${id}/preview`,
  
  // Graph endpoints
  GRAPHS: `${API_URL}/api/v1/graphs`,
  GRAPH_CREATE: `${API_URL}/api/v1/graphs/create`,
  
  // Query endpoints
  QUERY: `${API_URL}/api/v1/query`,
  BATCH_QUERY: `${API_URL}/api/v1/query/batch`,
  
  // Health check
  HEALTH: `${API_URL}/ping`,
};

import { useStore } from './store';
// Function to get headers with authentication token
export const getAuthHeaders = () => {
  const token = useStore.getState().auth.token;
  
  return {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    // Don't include Origin header as it can cause CORS issues
    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
  };
};
// Default request headers
export const DEFAULT_HEADERS = {
  'Content-Type': 'application/json',
};
// Function to get upload headers with authentication token
export const getUploadHeaders = () => {
  const token = useStore.getState().auth.token;
  
  return {
    'Accept': 'application/json',
    // Don't include Origin header as it can cause CORS issues
    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
  };
};
// File upload headers
export const UPLOAD_HEADERS = {
  'Accept': 'application/json',
};
// Request timeout in milliseconds
export const REQUEST_TIMEOUT = 30000;

// API functions
export const uploadFile = async (file: File): Promise<any> => {
  const formData = new FormData();
  formData.append('file', file);
  
  console.log('Uploading file to:', API_ENDPOINTS.DOCUMENT_UPLOAD);
  
  try {
    const response = await fetch(API_ENDPOINTS.DOCUMENT_UPLOAD, {
      method: 'POST',
      body: formData,
      credentials: 'include',
      headers: getUploadHeaders()
    });
    
    if (!response.ok) {
      throw new ApiError(`Failed to upload file: ${response.statusText}`, response.status);
    }
    
    return response.json();
  } catch (error) {
    console.error('Error uploading file:', error);
    throw error;
  }
};
export const uploadFiles = async (files: File[]): Promise<any> => {
  const formData = new FormData();
  
  // Append each file with the same field name
  files.forEach(file => {
    formData.append('files', file);
  });
  
  console.log(`Uploading ${files.length} files in batch to:`, API_ENDPOINTS.BATCH_DOCUMENT_UPLOAD);
  
  try {
    const response = await fetch(API_ENDPOINTS.BATCH_DOCUMENT_UPLOAD, {
      method: 'POST',
      body: formData,
      credentials: 'include',
      headers: getUploadHeaders()
    });
    
    if (!response.ok) {
      throw new ApiError(`Failed to upload files: ${response.statusText}`, response.status);
    }
    
    return response.json();
  } catch (error) {
    console.error('Error uploading files:', error);
    throw error;
  }
};
export const runQuery = async (row: any, column: any, globalRules: any = []): Promise<any> => {
  // Get document ID from the row's sourceData
  const documentId = row?.sourceData?.document?.id || "00000000000000000000000000000000";
  
  // Create a unique prompt ID if not provided
  const promptId = column?.id || Math.random().toString(36).substring(2, 15);
  
  // Combine column rules with global rules
  const rules = [...(column?.rules || []), ...(globalRules || [])];
  
  console.log('Running query to:', API_ENDPOINTS.QUERY);
  
  try {
    const response = await fetch(API_ENDPOINTS.QUERY, {
      method: 'POST',
      headers: getAuthHeaders(),
      credentials: 'include',
      body: JSON.stringify({
        document_id: documentId,
        prompt: {
          id: promptId,
          entity_type: column?.entityType || "",
          query: column?.query || "",
          type: column?.type || "str",
          rules: rules
        }
      }),
    });
    
    if (!response.ok) {
      throw new ApiError(`Query failed: ${response.statusText}`, response.status);
    }
    
    return response.json();
  } catch (error) {
    console.error('Error running query:', error);
    throw error;
  }
};
/**
 * Run multiple queries in parallel using the batch endpoint
 * @param queries Array of query objects, each containing row, column, and globalRules
 * @returns Array of query results in the same order as the input queries
 */
export const fetchDocumentPreview = async (documentId: string): Promise<string> => {
  console.log(`Fetching document preview for ID: ${documentId}`);
  console.log(`Preview URL: ${API_ENDPOINTS.DOCUMENT_PREVIEW(documentId)}`);
  
  try {
    const response = await fetch(API_ENDPOINTS.DOCUMENT_PREVIEW(documentId), {
      method: 'GET',
      headers: getAuthHeaders(),
      credentials: 'include',
    });
    
    console.log(`Preview response status: ${response.status} ${response.statusText}`);
    
    if (!response.ok) {
      console.error(`Error response from preview endpoint: ${response.status} ${response.statusText}`);
      throw new ApiError(`Failed to fetch document preview: ${response.statusText}`, response.status);
    }
    
    const data = await response.json();
    // console.log(`Preview response data:`, data);
    
    if (!data || !data.content) {
      console.error('No content in preview response:', data);
      return '';
    }
    
    return data.content;
  } catch (error) {
    console.error('Error fetching document preview:', error);
    throw error;
  }
};

// IMPROVED PROGRESS TRACKING SYSTEM
// Use a more reliable tracking mechanism with a Set to ensure each query is counted exactly once
let _processedIndices = new Set<number>();
let _totalQueries = 0;

// Reset tracking counters
function resetQueryTracking(total: number): void {
  _processedIndices.clear();
  _totalQueries = total;
  // console.log(`Reset query tracking: 0/${_totalQueries}`);
}

// Get current progress
function getQueryProgress(): { processed: number; total: number; percentage: number } {
  // Use the Set size as the source of truth for processed count
  const processed = _processedIndices.size;
  
  const percentage = _totalQueries > 0 ? Math.round((processed / _totalQueries) * 100) : 0;
  return {
    processed,
    total: _totalQueries,
    percentage
  };
}

// Update progress count safely
function updateProgressCount(index: number): number {
  // Only count each index once
  if (!_processedIndices.has(index)) {
    _processedIndices.add(index);
  }
  
  return _processedIndices.size;
}

/**
 * Advanced batch query executor with robust error handling and retry logic
 * 
 * This implementation uses a simplified, reliable approach to batch processing:
 * 1. It processes queries in small batches to ensure consistent UI updates
 * 2. It uses a strict tracking mechanism to ensure each query is processed exactly once
 * 3. It provides detailed logging for debugging and troubleshooting
 * 
 * @param queries Array of query objects, each containing row, column, and globalRules
 * @param options Configuration options for batch processing
 * @returns Promise resolving to an array of query results in the same order as input
 */
export const runBatchQueries = async (
  queries: Array<{ row: any; column: any; globalRules?: any[]; }>,
  options: {
    batchSize?: number;
    maxRetries?: number;
    retryDelay?: number;
    onBatchProgress?: (results: any[], batchIndex: number, totalBatches: number) => void;
    onQueryProgress?: (result: any, index: number, total: number) => void;
  } = {}
): Promise<any[]> => {
  // Generate a unique ID for this batch run to help with debugging
  const batchRunId = Math.random().toString(36).substring(2, 10);
  console.log(`[${batchRunId}] Starting batch query execution for ${queries.length} queries`);
  
  // Default options with optimized values
  const {
    batchSize = 5,  // Use smaller batches for more reliable processing
    maxRetries = 2,  // Keep retry count at 2
    retryDelay = 500,  // Keep retry delay at 500ms
    onBatchProgress,
    onQueryProgress
  } = options;
  
  // Reset tracking counters with exact count
  const exactQueryCount = queries.length;
  resetQueryTracking(exactQueryCount);
  // console.log(`[${batchRunId}] Reset query tracking: 0/${exactQueryCount}`);
  
  // Create a set to track which indices have been processed
  // This is our single source of truth for tracking progress
  const processedIndices = new Set<number>();
  
  // Function to safely update progress - this is the ONLY place where progress should be updated
  const updateProgress = (result: any, index: number, results: any[]) => {
    // Validate index is in range
    if (index < 0 || index >= results.length) {
      console.error(`[${batchRunId}] Invalid index ${index} (out of range 0-${results.length-1})`);
      return false;
    }
    
    // Only count each query once - strict check
    if (processedIndices.has(index)) {
      console.warn(`[${batchRunId}] Query ${index} already processed, skipping duplicate update`);
      return false;
    }
    
    // Update result
    results[index] = result;
    
    // Mark as processed
    processedIndices.add(index);
    
    // Update the module-level counter using the safe update function
    updateProgressCount(index);
    
    const progress = getQueryProgress();
    // Log accurate progress
    console.log(`[${batchRunId}] Progress: ${progress.processed}/${progress.total} (${progress.percentage}%)`);
    
    // Call progress callback if provided
    if (onQueryProgress) {
      try {
        // The callback expects (result, index, total) - pass our exact processed count
        onQueryProgress(result, index, progress.processed);
        
        // Log the exact progress values for debugging
        // console.log(`[${batchRunId}] Progress callback called with exact values: ${progress.processed}/${progress.total} (${progress.percentage}%)`);
      } catch (error) {
        console.error(`[${batchRunId}] Error in query progress callback for index ${index}:`, error);
      }
    }
    
    return true;
  };
  
  // Format the queries for the batch endpoint and ensure unique IDs
  const formattedQueries = queries.map(({ row, column, globalRules = [] }, index) => {
    const documentId = row?.sourceData?.document?.id || "00000000000000000000000000000000";
    const promptId = column?.id || Math.random().toString(36).substring(2, 15);
    const rules = [...(column?.rules || []), ...(globalRules || [])];
    
    // Calculate complexity score for prioritization
    let complexity = 0;
    
    // Inference queries (no document) are fastest
    if (documentId === "00000000000000000000000000000000") {
      complexity = 0;
    } 
    // Queries with rules are more complex
    else if (rules && rules.length > 0) {
      complexity = 2;
    }
    // Boolean queries often require more processing
    else if (column?.type === "bool") {
      complexity = 2;
    }
    // Array responses are more complex than simple types
    else if (column?.type?.includes("array")) {
      complexity = 1;
    }
    // Simple string/int queries
    else {
      complexity = 1;
    }
    
    return {
      document_id: documentId,
      prompt: {
        id: promptId,
        entity_type: column?.entityType || "",
        query: column?.query || "",
        type: column?.type || "str",
        rules: rules
      },
      // Store original query info for callbacks with guaranteed unique index
      _originalQuery: { 
        row, 
        column, 
        index: index,  // Use the array index directly to ensure uniqueness
        complexity: complexity,
        processed: false // Track if this query has been processed
      } 
    };
  });
  
  // If no queries, return empty array
  if (formattedQueries.length === 0) {
    console.warn("No queries to process, returning empty results");
    return [];
  }
  
  // Initialize results array with the exact length needed
  const results: any[] = new Array(formattedQueries.length).fill(null);
  
  // Group queries by row ID for row-based processing
  const queriesByRow = new Map<string, any[]>();
  
  // Group queries by row ID
  formattedQueries.forEach(query => {
    const rowId = query._originalQuery.row?.id || 'unknown';
    if (!queriesByRow.has(rowId)) {
      queriesByRow.set(rowId, []);
    }
    queriesByRow.get(rowId)!.push(query);
  });
  
  // Create simple fixed-size batches - no complex sorting or prioritization
  const batches = createBatches(formattedQueries, batchSize);
  
  // Log batch information
  console.log(`[${batchRunId}] Created ${batches.length} batches for processing ${formattedQueries.length} queries`);
  
  // Track failed queries for retry
  const failedQueries: { query: any; retryCount: number; }[] = [];
  
  try {
    // Process batches sequentially with detailed logging
    for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
      const batch = batches[batchIndex];
      console.log(`Processing batch ${batchIndex + 1}/${batches.length} with ${batch.length} queries`);
      
      // Log which rows are being processed in this batch
      // const rowIds = new Set(batch.map(q => q._originalQuery.row?.id));
      // console.log(`[${batchRunId}] Batch ${batchIndex + 1} contains queries for rows: ${Array.from(rowIds).join(', ')}`);
      
      try {
        // Process the batch
        const batchResults = await processBatch(batch);
        
        // Verify we got the expected number of results
        if (batchResults.length !== batch.length) {
          console.error(`[${batchRunId}] Batch result length mismatch: expected ${batch.length}, got ${batchResults.length}`);
        }
        
        // Call batch progress callback if provided
        if (onBatchProgress) {
          try {
            onBatchProgress(batchResults, batchIndex, batches.length);
          } catch (error) {
            console.error("Error in batch progress callback:", error);
          }
        }
        
        // Update individual query progress with detailed logging
        for (let i = 0; i < batchResults.length; i++) {
          if (i >= batch.length) {
            console.error(`Result index ${i} exceeds batch length ${batch.length}`);
            continue;
          }
          
          const originalIndex = batch[i]._originalQuery.index;
          // const rowId = batch[i]._originalQuery.row?.id;
          // const columnId = batch[i]._originalQuery.column?.id;
          
          // Log the update
          // console.log(`[${batchRunId}] Updating result for query ${originalIndex} (row: ${rowId}, column: ${columnId})`);
          
          // Mark query as processed
          batch[i]._originalQuery.processed = true;
          
          // Update progress and check if it was successful
          const updated = updateProgress(batchResults[i], originalIndex, results);
          if (!updated) {
            console.warn(`[${batchRunId}] Failed to update result for query ${originalIndex} - already processed?`);
          }
        }
        
        // Double-check that all queries in this batch are now processed
        const unprocessedQueries = batch.filter(q => !q._originalQuery.processed);
        if (unprocessedQueries.length > 0) {
          console.warn(`[${batchRunId}] ${unprocessedQueries.length} queries in batch ${batchIndex + 1} were not marked as processed`);
          
          // Log details about unprocessed queries for debugging
          unprocessedQueries.forEach(q => {
            console.warn(`[${batchRunId}] Unprocessed query: index=${q._originalQuery.index}, row=${q._originalQuery.row?.id}, column=${q._originalQuery.column?.id}`);
          });
        }
      } catch (error) {
        console.error(`Error processing batch ${batchIndex + 1}:`, error);
        
        // Add failed queries to retry list with detailed logging
        batch.forEach(query => {
          if (!query._originalQuery.processed) {
            console.log(`Adding query ${query._originalQuery.index} to retry list`);
            failedQueries.push({ query, retryCount: 0 });
          }
        });
      }
    }
  } catch (error) {
    console.error("Error in batch processing:", error);
  }
  
  // Process failed queries with retries
  if (failedQueries.length > 0) {
    console.log(`Processing ${failedQueries.length} failed queries with retries`);
    await processFailedQueries(failedQueries, results, maxRetries, retryDelay, onQueryProgress, processedIndices);
  }
  
  // Check for any missing results
  const missingIndices = [];
  for (let i = 0; i < results.length; i++) {
    if (!results[i]) {
      missingIndices.push(i);
    }
  }
  
  if (missingIndices.length > 0) {
    console.warn(`[${batchRunId}] Found ${missingIndices.length} missing results: ${missingIndices.join(', ')}`);
    
    // Process any missing queries individually as a last resort
    for (const index of missingIndices) {
      if (!processedIndices.has(index)) {
        const query = formattedQueries[index];
        console.log(`[${batchRunId}] Processing missing query ${index} individually`);
        
        try {
          const result = await processIndividualQuery(query);
          updateProgress(result, index, results);
        } catch (error) {
          console.error(`Error processing missing query ${index}:`, error);
          // Create fallback for this query
          const fallback = createFallbackResult(query);
          updateProgress(fallback, index, results);
        }
      }
    }
  }
  
  // Final verification of processed count
  if (processedIndices.size !== exactQueryCount) {
    console.error(`[${batchRunId}] Final processed count (${processedIndices.size}) doesn't match total (${exactQueryCount})`);
    
    // Log which indices were never processed
    const allIndices = new Set(Array.from({ length: exactQueryCount }, (_, i) => i));
    const unprocessedIndices = Array.from(allIndices).filter(i => !processedIndices.has(i));
    
    if (unprocessedIndices.length > 0) {
      console.error(`[${batchRunId}] Indices never processed: ${unprocessedIndices.join(', ')}`);
    }
    
    // Log any indices processed that weren't in the original set
    const extraIndices = Array.from(processedIndices).filter(i => i >= exactQueryCount);
    if (extraIndices.length > 0) {
      console.error(`[${batchRunId}] Extra indices processed: ${extraIndices.join(', ')}`);
    }
  }
  
  // Fill any remaining missing results with fallbacks
  fillMissingResults(formattedQueries, results, processedIndices);
  
  console.log(`[${batchRunId}] Batch query execution completed with ${processedIndices.size}/${exactQueryCount} queries processed`);
  
  return results;
};

/**
 * Process failed queries with retry logic
 */
async function processFailedQueries(
  failedQueries: { query: any; retryCount: number; }[],
  results: any[],
  maxRetries: number,
  retryDelay: number,
  onQueryProgress?: (result: any, index: number, total: number) => void,
  processedIndices?: Set<number>
): Promise<void> {
  // Process in smaller batches to avoid overwhelming the server
  const batchSize = 5;
  const batches = createBatches(failedQueries, batchSize);
  
  for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
    const batch = batches[batchIndex];
    console.log(`Processing retry batch ${batchIndex + 1}/${batches.length} with ${batch.length} queries`);
    
    // Process each query in the batch
    const batchPromises = batch.map(async ({ query, retryCount }) => {
      // Skip if we've exceeded max retries
      if (retryCount >= maxRetries) {
        console.log(`Query ${query._originalQuery.index} exceeded max retries`);
        return;
      }
      
      // Add jitter to retry delay to prevent thundering herd
      const jitter = Math.random() * 0.5 + 0.75; // 0.75-1.25
      const delay = retryDelay * jitter * (retryCount + 1);
      
      // Wait before retrying
      await new Promise(resolve => setTimeout(resolve, delay));
      
      try {
        // Process the query individually
        console.log(`Retry attempt ${retryCount + 1}/${maxRetries} for query ${query._originalQuery.index}`);
        const result = await processIndividualQuery(query);
        
        // Update progress safely
        const originalIndex = query._originalQuery.index;
        
        // Only update if not already processed
        if (originalIndex >= 0 && originalIndex < results.length && 
            (!processedIndices || !processedIndices.has(originalIndex))) {
          
          // Update result
          results[originalIndex] = result;
          
          // Mark as processed
          if (processedIndices) {
            processedIndices.add(originalIndex);
            updateProgressCount(originalIndex);
          }
          
          // Call progress callback if provided
          if (onQueryProgress) {
            try {
              // The callback expects (result, index, total) - pass our exact processed count
              const progress = getQueryProgress();
              onQueryProgress(result, originalIndex, progress.processed);
            } catch (error) {
              console.error(`Error in query progress callback for index ${originalIndex}:`, error);
            }
          }
        } else {
          console.log(`Skipping update for already processed query ${originalIndex}`);
        }
        
        return { success: true, query };
      } catch (error) {
        console.error(`Retry attempt ${retryCount + 1} failed for query ${query._originalQuery.index}:`, error);
        
        // If we haven't exceeded max retries, add back to failed queries with incremented retry count
        if (retryCount + 1 < maxRetries) {
          return { success: false, query, retryCount: retryCount + 1 };
        }
        
        // Create fallback result
        const fallbackResult = createFallbackResult(query);
        const originalIndex = query._originalQuery.index;
        
        // Only update if not already processed
        if (originalIndex >= 0 && originalIndex < results.length && 
            (!processedIndices || !processedIndices.has(originalIndex))) {
          
          // Update result with fallback
          results[originalIndex] = fallbackResult;
          
          // Mark as processed
          if (processedIndices) {
            processedIndices.add(originalIndex);
            updateProgressCount(originalIndex);
          }
          
          // Call progress callback if provided
          if (onQueryProgress) {
            try {
              // The callback expects (result, index, total) - pass our exact processed count
              const progress = getQueryProgress();
              onQueryProgress(fallbackResult, originalIndex, progress.processed);
            } catch (error) {
              console.error(`Error in query progress callback for index ${originalIndex}:`, error);
            }
          }
        } else {
          console.log(`Skipping fallback for already processed query ${originalIndex}`);
        }
        
        return { success: true, query }; // Mark as "success" to remove from retry queue
      }
    });
    
    // Wait for all queries in this batch to complete
    const batchResults = await Promise.all(batchPromises);
    
    // Collect queries that still need retrying
    const queriesToRetry = batchResults
      .filter((result): result is { success: false; query: any; retryCount: number } => 
        result !== undefined && !result.success)
      .map(result => ({ query: result.query, retryCount: result.retryCount }));
    
    // Add queries that need retrying to the next batch
    if (queriesToRetry.length > 0) {
      console.log(`Adding ${queriesToRetry.length} queries to retry queue`);
      failedQueries.push(...queriesToRetry);
    }
  }
}

/**
 * Process a single query
 */
async function processIndividualQuery(query: any): Promise<any> {
  console.log(`Processing individual query for index ${query._originalQuery.index}`);
  
  const startTime = performance.now();
  
  const response = await fetch(API_ENDPOINTS.QUERY, {
    method: 'POST',
    headers: getAuthHeaders(),
    credentials: 'include',
    mode: 'cors',
    body: JSON.stringify({
      document_id: query.document_id,
      prompt: query.prompt
    }),
    signal: AbortSignal.timeout(45000) // 45 second timeout for individual queries
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    console.error(`Individual query failed with status ${response.status}:`, errorText);
    throw new ApiError(`Query failed: ${response.statusText}`, response.status);
  }
  
  const result = await response.json();
  
  const totalTime = performance.now() - startTime;
  console.log(`Individual query completed in ${Math.round(totalTime)}ms`);
  
  return result;
}

/**
 * Process a batch of queries using the batch API endpoint with optimized timeout
 */
async function processBatch(batch: any[]): Promise<any[]> {
  // Optimized timeout calculation - faster for smaller batches
  // Base: 20s + 3s per query, capped at 45s (reduced from 60s)
  const timeoutMs = Math.min(45000, 20000 + (batch.length * 3000));
  
  console.log(`Processing batch of ${batch.length} queries with ${timeoutMs}ms timeout`);
  
  // Add performance tracking
  const startTime = performance.now();
  
  try {
    // Use keep-alive connection for better performance
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    
    const response = await fetch(API_ENDPOINTS.BATCH_QUERY, {
      method: 'POST',
      headers: {
        ...getAuthHeaders(),
        'Connection': 'keep-alive',
      },
      credentials: 'include',
      mode: 'cors',
      body: JSON.stringify(batch.map(q => ({ 
        document_id: q.document_id, 
        prompt: q.prompt 
      }))),
      signal: controller.signal
    });
    
    // Clear timeout
    clearTimeout(timeoutId);
    
    // Log performance metrics
    const networkTime = performance.now() - startTime;
    console.log(`Batch query network time: ${Math.round(networkTime)}ms`);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Batch query failed with status ${response.status}:`, errorText);
      throw new ApiError(`Batch query failed: ${response.statusText}`, response.status);
    }
    
    // Use streaming JSON parser for large responses if available
    const batchResults = await response.json();
    
    // Log total time including JSON parsing
    const totalTime = performance.now() - startTime;
    console.log(`Batch query total time: ${Math.round(totalTime)}ms`);
    
    if (!Array.isArray(batchResults)) {
      console.error('Batch results is not an array:', batchResults);
      throw new Error('Invalid batch response format');
    }
    
    return batchResults;
  } catch (error) {
    // Handle timeout errors specifically
    if (error instanceof Error && error.name === 'AbortError') {
      console.error(`Batch query timed out after ${timeoutMs}ms`);
      throw new Error(`Batch query timed out after ${timeoutMs}ms`);
    }
    
    // Log detailed error information
    console.error(`Batch query failed after ${Math.round(performance.now() - startTime)}ms:`, error);
    
    // Rethrow with more context
    if (error instanceof Error) {
      throw new Error(`Batch query failed: ${error.message} (batch size: ${batch.length})`);
    }
    throw error;
  }
}

/**
 * Create a fallback result for a failed query
 */
function createFallbackResult(query: any): any {
  return {
    answer: {
      id: `fallback-${Math.random().toString(36).substring(2)}`,
      document_id: query.document_id,
      prompt_id: query.prompt.id,
      answer: query.prompt.type === 'bool' ? false :
             query.prompt.type === 'int' ? 0 :
             query.prompt.type.includes('array') ? [] : 
             "Query failed to process",
      type: query.prompt.type
    },
    chunks: [],
    resolved_entities: []
  };
}

/**
 * Create batches from an array of queries
 */
function createBatches<T>(items: T[], batchSize: number): T[][] {
  const batches: T[][] = [];
  for (let i = 0; i < items.length; i += batchSize) {
    batches.push(items.slice(i, i + batchSize));
  }
  return batches;
}

/**
 * Fill any missing results with fallbacks and ensure UI updates
 */
function fillMissingResults(
  queries: any[], 
  results: any[], 
  processedIndices?: Set<number>
): void {
  const runId = Math.random().toString(36).substring(2, 6);
  let missingCount = 0;
  
  for (let i = 0; i < queries.length; i++) {
    if (!results[i]) {
      const query = queries[i];
      missingCount++;
      
      // Create fallback based on prompt type
      results[i] = createFallbackResult(query);
      
      // Mark as processed but only if not already processed
      if (processedIndices && !processedIndices.has(i)) {
        processedIndices.add(i);
        // Only increment the counter if we actually added to the set
        updateProgressCount(i);
      }
    }
  }
  
  if (missingCount > 0) {
    const progress = getQueryProgress();
    console.log(`[${runId}] Created ${missingCount} fallbacks for missing results`);
    console.log(`[${runId}] Final progress: ${progress.processed}/${progress.total} (${progress.percentage}%)`);
  }
  
  // Final verification
  const stillMissing = results.filter(r => !r).length;
  if (stillMissing > 0) {
    console.error(`[${runId}] CRITICAL ERROR: Still have ${stillMissing} missing results after fallback creation!`);
  } else {
    console.log(`[${runId}] All ${results.length} queries have results (including fallbacks)`);
  }
}

// Run a query against the API
export const runQuery = async (documentId: string, prompt: any): Promise<any> => {
  console.log(`Running query for document: ${documentId}, prompt: ${prompt.query}`);
  
  // Check document status first to avoid premature queries
  try {
    const statusResponse = await checkDocumentStatus(documentId);
    if (statusResponse.status === 'processing') {
      console.log(`Document ${documentId} is still processing, waiting before running query`);
      
      // Return a processing status response to inform the UI
      return {
        result: {
          answer: "Document is still processing, please wait...",
          processing: true
        },
        document_id: documentId,
        retrieval_type: prompt.type,
        query: prompt.query
      };
    }
  } catch (statusError) {
    console.warn('Error checking document status before query:', statusError);
    // Continue with query anyway if status check fails
  }
  
  try {
    // Prepare the request payload
    const payload = {
      document_id: documentId,
      prompt: prompt
    };
    
    console.log(`Query payload:`, payload);
    
    // Execute the query
    const response = await fetch(API_ENDPOINTS.QUERY, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(payload),
      credentials: 'include',
      mode: 'cors'
    });
    
    if (!response.ok) {
      console.error(`Failed to run query: ${response.statusText}`);
      throw new ApiError(`Failed to run query: ${response.statusText}`, response.status);
    }
    
    const data = await response.json();
    console.log(`Query response:`, data);
    
    // Check if the document is still processing
    if (data.result && data.result.processing) {
      console.log(`Document ${documentId} is still processing during query`);
      
      // Set up to retry after a delay
      await new Promise(resolve => setTimeout(resolve, 2000));
      return runQuery(documentId, prompt); // Recursive retry
    }
    
    return data;
  } catch (error) {
    console.error('Error running query:', error);
    throw error;
  }
}; 
import { useStore } from "../config/store";
import { documentStatusCache } from "../services/document-status-cache";

/**
 * Updates document statuses in the table state
 * This function finds all documents marked as "processing" in the table state
 * and checks their actual status from the backend. If the status has changed,
 * it updates the table state and saves it back to the backend.
 */
export async function updateDocumentStatusesInTableState(retryCount = 0, maxRetries = 3) {
  const store = useStore.getState();
  const activeTableId = store.activeTableId;
  
  if (!activeTableId) return;
  
  // Get the current table
  const table = store.getTable(activeTableId);
  if (!table) return;
  
  // Find all rows with documents, not just processing ones
  const documentRows = table.rows.filter(
    row => row.sourceData &&
           row.sourceData.type === 'document' &&
           row.sourceData.document
  );
  
  if (documentRows.length === 0) return;
  
  console.log(`Found ${documentRows.length} documents, checking actual status...`);
  
  // Check actual status for each document
  let tableUpdated = false;
  
  for (const row of documentRows) {
    // We've already filtered for rows with document source data
    const sourceData = row.sourceData!;
    const document = (sourceData as { type: 'document', document: any }).document;
    const docId = document.id;
    
    try {
      // Check actual status from backend via cache
      const status = await documentStatusCache.getStatus(docId);
      
      // Always update the status to match the backend, regardless of current status
      if (status !== document.status) {
        console.log(`Updating document ${docId} status from ${document.status} to ${status}`);
        
        // Create updated rows with the new document status
        const updatedRows = table.rows.map(r => {
          if (r.id === row.id) {
            return {
              ...r,
              sourceData: {
                ...sourceData,
                document: {
                  ...document,
                  status: status
                }
              }
            };
          }
          return r;
        });
        
        // Update the table with the new rows
        store.editTable(activeTableId, {
          rows: updatedRows
        });
        
        // Also update the status in the documents store
        if (store.documents[docId]) {
          store.updateDocumentStatus(docId, status);
        }
        
        tableUpdated = true;
      }
    } catch (error) {
      console.error(`Error checking status for document ${docId}:`, error);
      
      // Default to completed on error
      const updatedRows = table.rows.map(r => {
        if (r.id === row.id) {
          return {
            ...r,
            sourceData: {
              ...sourceData,
              document: {
                ...document,
                status: 'completed'
              }
            }
          };
        }
        return r;
      });
      
      // Update the table with the new rows
      store.editTable(activeTableId, {
        rows: updatedRows
      });
      
      // Also update the status in the documents store
      if (store.documents[docId]) {
        store.updateDocumentStatus(docId, 'completed');
      }
      
      tableUpdated = true;
    }
  }
  
  // Debug: Log document statuses before update
  console.log('Document statuses before update:',
    table.rows
      .filter(row => row.sourceData?.type === 'document')
      .map(row => {
        // Type assertion to handle TypeScript errors
        const docData = row.sourceData as { type: 'document', document: { id: string, status: string } };
        return {
          id: docData.document.id,
          status: docData.document.status
        };
      })
  );

  // Save the updated table state if any changes were made
  if (tableUpdated) {
    console.log('Saving updated table state with corrected document statuses');
    
    try {
      // Force an immediate save
      await store.saveTableState();
      
      // Force a reload of the table state to verify changes
      console.log('Reloading table state to verify changes');
      await store.loadLatestTableState();
      
      // Debug: Log document statuses after update
      const updatedRows = store.getTable(activeTableId).rows
        .filter(row => row.sourceData?.type === 'document')
        .map(row => {
          // Type assertion to handle TypeScript errors
          const docData = row.sourceData as { type: 'document', document: { id: string, status: string } };
          return {
            id: docData.document.id,
            status: docData.document.status
          };
        });
      
      console.log('Document statuses after update:', updatedRows);
      
      // Check if any documents are still marked as processing but should be completed
      const stillProcessing = updatedRows.filter(row => row.status === 'processing');
      
      if (stillProcessing.length > 0 && retryCount < maxRetries) {
        console.log(`Found ${stillProcessing.length} documents still marked as processing. Retrying update (${retryCount + 1}/${maxRetries})...`);
        
        // Wait a bit before retrying
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Retry the update
        return updateDocumentStatusesInTableState(retryCount + 1, maxRetries);
      }
    } catch (error) {
      console.error('Error saving or reloading table state:', error);
      
      if (retryCount < maxRetries) {
        console.log(`Error occurred during update. Retrying (${retryCount + 1}/${maxRetries})...`);
        
        // Wait a bit before retrying
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Retry the update
        return updateDocumentStatusesInTableState(retryCount + 1, maxRetries);
      }
    }
  }
}
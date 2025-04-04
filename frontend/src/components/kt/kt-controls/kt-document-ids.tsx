import { useState } from "react";
import { 
  Button, 
  Modal, 
  Text, 
  Group, 
  Select, 
  Stack
} from "@mantine/core";
import { IconRefreshDot } from "@tabler/icons-react";
import { useStore } from "@config/store";
import { notifications } from "@utils/notifications";
import { CellValue } from "@config/store/store.types";

export function KtDocumentIds() {
  const [opened, setOpened] = useState(false);
  const [selectedColumn, setSelectedColumn] = useState<string | null>(null);
  
  const table = useStore(store => store.getTable());
  const columns = table.columns.filter(col => !col.hidden);
  const rows = table.rows.filter(row => !row.hidden);
  
  // Filter out columns that don't have an entity type (empty columns)
  const columnOptions = columns
    .filter(column => column.entityType && column.entityType.trim() !== '')
    .map(column => ({
      value: column.id,
      label: column.entityType,
    }));
  
  // Start processing document IDs - now just extracts data and initiates processing
  const handleProcessIds = async () => {
    if (!selectedColumn) return;
    
    try {
      // Update table state to indicate document processing is starting
      // Similar to how batch queries show progress
      setOpened(false); // Close the modal
      
      // Extract document IDs from the selected column
      const documentIds: string[] = [];
      const docIdToRowMap = new Map<string, string>();
      
      for (const row of rows) {
        const cell: CellValue = row.cells[selectedColumn];
        // Handle different cell value types
        let cellValue: string | null = null;
        
        if (typeof cell === 'string') {
          cellValue = cell;
        } else if (typeof cell === 'object' && cell && 'text' in cell) {
          cellValue = cell.text;
        }
        
        if (cellValue && cellValue.trim()) {
          const docId = cellValue.trim();
          // Store document ID and its row ID mapping
          documentIds.push(docId);
          docIdToRowMap.set(docId, row.id);
        }
      }
      
      if (documentIds.length === 0) {
        notifications.show({
          message: 'No document IDs found in the selected column',
          color: 'red'
        });
        return;
      }
      
      // Set the table state to indicate document processing is in progress
      // and update the progress indicator in the toolbar
      useStore.getState().editActiveTable({
        requestProgress: {
          total: documentIds.length,
          completed: 0,
          inProgress: true
        }
      });
      
      notifications.show({
        message: `Processing ${documentIds.length} document IDs...`,
        color: 'blue'
      });
      
      // Start the processing in the background using the store action
      processDocumentIdsWithStore(documentIds, docIdToRowMap, selectedColumn);
      
    } catch (error) {
      console.error('Error initiating document ID processing:', error);
      notifications.show({
        message: 'Error initiating document ID processing',
        color: 'red'
      });
      
      // Reset progress state on error
      useStore.getState().editActiveTable({
        requestProgress: {
          total: 0,
          completed: 0,
          inProgress: false,
          error: true
        }
      });
    }
  };

  // Renamed function to reflect it uses the store action
  const processDocumentIdsWithStore = async (
    documentIds: string[],
    docIdToRowMap: Map<string, string>,
    selectedColumn: string
  ) => {
    const { ingestSingleDocumentById, editActiveTable, getTable } = useStore.getState();
    let successCount = 0;
    const totalToIngest = documentIds.length;

    try {
      // Process documents sequentially using the store action
      // Parallel processing can be added later if performance is an issue
      for (const documentId of documentIds) {
        const rowId = docIdToRowMap.get(documentId);
        if (!rowId) {
          console.warn(`Row ID not found for document ID: ${documentId}, skipping.`);
          continue; // Skip if row mapping is lost
        }

        // Call the store action to process this single ID
        const result = await ingestSingleDocumentById(documentId, rowId, selectedColumn);
        if (result.success) {
          successCount++;
        }

        // Update progress in the store after each attempt
        const currentProgress = getTable().requestProgress || { total: totalToIngest, completed: 0, inProgress: true };
        editActiveTable({
          requestProgress: {
            ...currentProgress,
            // Increment completed count, ensuring it doesn't exceed total
            completed: Math.min(currentProgress.completed + 1, totalToIngest) 
          }
        });
      }

      // --- Finalization ---
      // Set the final progress state AFTER all documents are processed
      editActiveTable({
        requestProgress: {
          total: totalToIngest,
          completed: totalToIngest, // Ensure it shows 100% completion visually
          inProgress: false, // Mark as no longer in progress
          error: successCount < totalToIngest // Indicate error if any failed
        }
      });

      // Show ONE summary notification at the end
      if (successCount === totalToIngest && totalToIngest > 0) {
        notifications.show({
          title: 'Ingestion Complete',
          message: `Successfully processed all ${totalToIngest} document IDs.`,
          color: 'green'
        });
      } else if (successCount > 0 && successCount < totalToIngest) {
        notifications.show({
          title: 'Ingestion Partially Complete',
          message: `Successfully processed ${successCount} of ${totalToIngest} document IDs.`,
          color: 'orange'
        });
      } else if (successCount === 0 && totalToIngest > 0) {
        notifications.show({
          title: 'Ingestion Failed',
          message: `Failed to process any of the ${totalToIngest} document IDs. Check console for errors.`,
          color: 'red'
        });
      }
      // No notification if totalToIngest was 0 initially

    } catch (error) {
      console.error('Error during bulk document ID processing:', error);
      notifications.show({
        title: 'Processing Error',
        message: 'An unexpected error occurred during document processing.',
        color: 'red'
      });

      // Ensure progress state is updated to show error and stop progress
      editActiveTable({
        requestProgress: {
          total: totalToIngest,
          completed: successCount, // Show how many completed before error
          inProgress: false,
          error: true
        }
      });
    } finally {
       // Ensure progress is always marked as not in progress in case of unexpected exit
       const currentProgress = getTable().requestProgress;
       if (currentProgress?.inProgress) {
         editActiveTable({
           requestProgress: { ...currentProgress, inProgress: false }
         });
       }
    }
  };

  return (
    <>
      <Button
        leftSection={<IconRefreshDot size={14} />}
        size="sm"
        onClick={() => setOpened(true)}
      >
        Ingest Documents (SFID)
      </Button>
      
      <Modal 
        opened={opened} 
        onClose={() => setOpened(false)} 
        title="Ingest Documents from Salesforce IDs"
        size="md"
      >
        <Stack>
          <Text size="sm">
            Select a column that contains Salesforce document IDs. The system will fetch the document content
            for each ID and process it for use in the RAG pipeline.
          </Text>
          
          <Select
            label="Select column with document IDs"
            placeholder="Choose a column"
            value={selectedColumn}
            onChange={setSelectedColumn}
            data={columnOptions}
            required
          />
          
          <Group justify="flex-end" mt="md">
            <Button variant="default" onClick={() => setOpened(false)}>
              Cancel
            </Button>
            <Button 
              onClick={handleProcessIds} 
              disabled={!selectedColumn}
            >
              Ingest Documents
            </Button>
          </Group>
        </Stack>
      </Modal>
    </>
  );
}

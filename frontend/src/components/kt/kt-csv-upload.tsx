import { useState } from "react";
import { Button, Group, FileButton, Tooltip, Modal, Text } from "@mantine/core";
import { IconTable } from "@tabler/icons-react";
import { useStore } from "@config/store";
import { getBlankColumn, getBlankRow } from "@config/store/store.utils";
import { AnswerTableRow } from "@config/store/store.types";
import { notifications } from "@utils/notifications";

export function KtCsvUpload() {
  const [loading, setLoading] = useState(false);
  const [confirmModalOpen, setConfirmModalOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFileSelect = (file: File | null) => {
    if (!file) return;
    
    // Check if file is CSV
    const fileExtension = file.name.split('.').pop()?.toLowerCase();
    if (fileExtension !== 'csv') {
      notifications.show({
        title: 'Invalid file type',
        message: 'Please upload a CSV file',
        color: 'red'
      });
      return;
    }
    
    // Store the selected file and open confirmation modal
    setSelectedFile(file);
    setConfirmModalOpen(true);
  };
  
  const handleCsvUpload = async () => {
    if (!selectedFile) return;
    
    setLoading(true);
    setConfirmModalOpen(false);
    
    try {
      // Read the file
      const reader = new FileReader();
      
      reader.onload = async (e) => {
        try {
          const content = e.target?.result as string;
          
          // Parse CSV content
          const data = parseCSV(content);
          
          // console.log("CSV data to import:", data);
          
          // Import data directly into the grid
          try {
            const store = useStore.getState();
            const isAuthenticated = store.auth.isAuthenticated;
            
            // Get header row (first row of CSV)
            const headerRow = data[0];
            const dataRows = data.slice(1);
            
            // Create new columns based on header row
            const newColumns = headerRow.map((header) => {
              // Create a new column for each header
              return {
                ...getBlankColumn(),
                entityType: header || 'Column',
                query: header || 'Column', 
                generate: false, // Don't auto-generate for imported data
                type: "str" as const, // Default to string type for CSV data
                hidden: false // Make sure columns are not hidden
              };
            });
            
            // Create rows with cell data
            const newRows: AnswerTableRow[] = dataRows.map((rowData) => {
              // Create a blank row
              const row = getBlankRow();
              
              // Add cell data
              rowData.forEach((cellValue, colIndex) => {
                if (colIndex < newColumns.length) {
                  // Check if the value is undefined or null first
                  if (cellValue === null || cellValue === undefined) {
                    row.cells[newColumns[colIndex].id] = '';
                    return;
                  }
                  
                  // Make sure we're working with a string
                  let value = String(cellValue);
                  
                  // If it's a single quote character, replace with empty string
                  if (value === '"') {
                    row.cells[newColumns[colIndex].id] = '';
                    return;
                  }
                  
                  // If it's just whitespace, preserve it
                  if (value.trim() === '') {
                    row.cells[newColumns[colIndex].id] = value;
                    return;
                  }
                  
                  // Handle quoted empty strings (like "")
                  if (value === '""') {
                    row.cells[newColumns[colIndex].id] = '';
                    return;
                  }
                  
                  // Remove surrounding quotes if they exist
                  if (value.startsWith('"') && value.endsWith('"') && value.length >= 2) {
                    value = value.substring(1, value.length - 1);
                  }
                  
                  row.cells[newColumns[colIndex].id] = value;
                }
              });
              
              row.sourceData = null;
              
              return row;
            });
            
            // Replace existing columns and rows with new ones
            store.editActiveTable({
              columns: newColumns,
              rows: newRows
            });
            
            // Force a UI refresh and save to server if authenticated
            try {
              await store.saveTableState();
              
              console.log("CSV import successful");
              // console.log("New columns:", newColumns);
              // console.log("New rows:", newRows);
              // console.log("Current table state:", store.getTable());
              
              notifications.show({
                title: 'CSV imported',
                message: `Successfully imported data from ${selectedFile.name}`,
                color: 'green'
              });
            } catch (saveError) {
              console.error("Error saving table state:", saveError);
              throw new Error(
                isAuthenticated 
                  ? "Error saving data to server. Please try again." 
                  : "Error saving data to browser storage. The file may be too large. Try logging in to use server storage."
              );
            }
          } catch (error) {
            console.error("Error importing CSV data:", error);
            
            // Check if the error is related to storage quota
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            if (errorMessage.includes('quota') || errorMessage.includes('storage')) {
              throw new Error(
                "Failed to save data: Browser storage limit exceeded. " +
                "Please try a smaller file."
              );
            }
            
            throw error;
          }
        } catch (error) {
          console.error('Error processing CSV file:', error);
          notifications.show({
            title: 'Import failed',
            message: error instanceof Error ? error.message : 'Unknown error',
            color: 'red'
          });
        } finally {
          setLoading(false);
        }
      };
      
      reader.onerror = () => {
        notifications.show({
          title: 'Read failed',
          message: 'Failed to read the file',
          color: 'red'
        });
        setLoading(false);
      };
      
      reader.readAsText(selectedFile);
    } catch (error) {
      console.error('Error uploading CSV file:', error);
      notifications.show({
        title: 'Upload failed',
        message: error instanceof Error ? error.message : 'Unknown error',
        color: 'red'
      });
      setLoading(false);
    }
  };
  
  // CSV parser function that handles multi-line content in cells
  const parseCSV = (text: string): string[][] => {
    const result: string[][] = [];
    let row: string[] = [];
    let inQuote = false;
    let currentValue = '';
    
    // Process character by character
    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      const nextChar = text[i + 1] || '';
      
      // Handle quotes
      if (char === '"') {
        // Check for escaped quotes (double quotes)
        if (nextChar === '"') {
          currentValue += '"';
          i++; // Skip the next quote
        } else {
          // Toggle quote state
          inQuote = !inQuote;
        }
      }
      // Handle commas (only if not in quotes)
      else if (char === ',' && !inQuote) {
        row.push(currentValue);
        currentValue = '';
      }
      // Handle newlines (only if not in quotes)
      else if ((char === '\n' || (char === '\r' && nextChar === '\n')) && !inQuote) {
        // Add the last value to the row
        row.push(currentValue);
        
        // Add the row to the result if it's not empty
        if (row.some(cell => cell.trim() !== '')) {
          result.push(row);
        }
        
        // Reset for next row
        row = [];
        currentValue = '';
        
        // Skip the \n if we just processed \r
        if (char === '\r') {
          i++;
        }
      }
      // All other characters
      else {
        currentValue += char;
      }
    }
    
    // Add the last value and row if there's any
    if (currentValue !== '' || row.length > 0) {
      row.push(currentValue);
      result.push(row);
    }
    
    // Post-process to clean up the data
    // This helps handle edge cases with quotes and empty cells
    return result.map(row => 
      row.map(cell => {
        // Special handling for single quotes
        if (cell === '"') return '';
        
        // Handle whitespace-only cells (preserve them)
        if (cell.trim() === '' && cell !== '') return cell;
        
        // Handle quoted empty strings (like "")
        if (cell === '""') return '';
        
        // Some CSV formats encode a blank cell as a single quote character
        // Check for this specific pattern
        if (cell.trim() === '"') return '';
        
        // Handle quoted strings - remove surrounding quotes
        if (cell.startsWith('"') && cell.endsWith('"') && cell.length >= 2) {
          return cell.substring(1, cell.length - 1);
        }
        
        return cell;
      })
    );
  };

  return (
    <>
      <Modal 
        opened={confirmModalOpen} 
        onClose={() => setConfirmModalOpen(false)}
        title="Confirm CSV Import"
        centered
      >
        <Text mb="md">
          This will import data from the CSV file into the grid.
          The first row will be used as column headers, and the remaining rows as data.
        </Text>
        <Text mb="md">
          Note: Multi-line content in cells will be preserved.
        </Text>
        {selectedFile && selectedFile.size > 1000000 && !useStore.getState().auth.isAuthenticated && (
          <Text mb="md" color="orange" fw={500}>
            Warning: This is a large CSV file ({(selectedFile.size / (1024 * 1024)).toFixed(2)} MB).
          </Text>
        )}
        <Text mb="xl" fw={500}>
          Are you sure you want to continue?
        </Text>
        <Group justify="flex-end">
          <Button variant="outline" onClick={() => setConfirmModalOpen(false)}>
            Cancel
          </Button>
          <Button color="blue" onClick={handleCsvUpload} loading={loading}>
            Import CSV
          </Button>
        </Group>
      </Modal>
      
      <Tooltip label="Import CSV data">
        <Group>
          <FileButton onChange={handleFileSelect} accept=".csv">
            {(props) => (
              <Button
                {...props}
                variant="outline"
                leftSection={<IconTable size={16} />}
                color="blue"
              >
                Import CSV
              </Button>
            )}
          </FileButton>
        </Group>
      </Tooltip>
    </>
  );
}

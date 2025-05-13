import { create } from "zustand";
import { persist } from "zustand/middleware";
import {
  castArray,
  cloneDeep,
  compact,
  fromPairs,
  groupBy,
  isArray,
  isEmpty,
  isNil,
  keyBy,
  mapValues,
  omit
} from "lodash-es";
import cuid from "@bugsnag/cuid";
import {
  getBlankColumn,
  getBlankRow,
  getBlankTable,
  getCellKey,
  getInitialData,
  isArrayType,
  toSingleType
} from "./store.utils";
import { AnswerTableRow, ResolvedEntity, SourceData, Store } from "./store.types";
// Import API_ENDPOINTS
import { runBatchQueries, uploadFile, API_ENDPOINTS, checkDocumentStatus } from "../api";
import { AuthError, login as apiLogin, verifyToken } from "../../services/api/auth";
import { 
  saveTableState as apiSaveTableState, 
  updateTableState as apiUpdateTableState,
  listTableStates
} from "../../services/api/table-state";
import { notifications } from "../../utils/notifications";
import { insertAfter, insertBefore, where } from "@utils/functions";

export const useStore = create<Store>()(
  persist(
    (set, get) => ({
      colorScheme: "light",
      ...getInitialData(),
      activePopoverId: null,
      documentPreviews: {}, // Initialize empty document previews
      documents: {}, // Track uploaded documents and their statuses
      isLoading: false, // Add isLoading to initial state
      auth: {
        token: null,
        isAuthenticated: false,
        isAuthenticating: false
      },
      
      // Authentication methods
      login: async (password: string) => {
        try {
          set({ auth: { ...get().auth, isAuthenticating: true } });
          
          const response = await apiLogin(password);
          
          set({
            auth: {
              token: response.access_token,
              isAuthenticated: true,
              isAuthenticating: false
            }
          });
          
          notifications.show({
            title: 'Login successful',
            message: 'You have been successfully authenticated',
            color: 'green'
          });
        } catch (error) {
          set({
            auth: {
              ...get().auth,
              isAuthenticating: false
            }
          });
          
          if (error instanceof AuthError) {
            notifications.show({
              title: 'Authentication failed',
              message: error.message,
              color: 'red'
            });
          } else {
            notifications.show({
              title: 'Authentication failed',
              message: error instanceof Error ? error.message : 'Unknown error',
              color: 'red'
            });
          }
          
          throw error;
        }
      },
      
      logout: () => {
        set({
          auth: {
            token: null,
            isAuthenticated: false,
            isAuthenticating: false
          }
        });
        
        notifications.show({
          title: 'Logged out',
          message: 'You have been logged out',
          color: 'blue'
        });
      },
      
      checkAuth: async () => {
        const { auth } = get();
        
        if (!auth.token) {
          return false;
        }
        
        try {
          const result = await verifyToken(auth.token);
          
          if (!result || !result.isValid) {
            set({
              auth: {
                token: null,
                isAuthenticated: false,
                isAuthenticating: false
              }
            });
            return false;
          } else if (!auth.isAuthenticated) {
            set({
              auth: {
                ...auth,
                isAuthenticated: true
              }
            });
          }
          
          return result;
        } catch (error) {
          console.error('Error verifying token:', error);
          
          set({
            auth: {
              token: null,
              isAuthenticated: false,
              isAuthenticating: false
            }
          });
          
          return false;
        }
      },

      toggleColorScheme: () => {
        set({ colorScheme: get().colorScheme === "light" ? "dark" : "light" });
      },
      
      setActivePopover: (id: string | null) => {
        set({ activePopoverId: id });
      },

      // Add document preview content to the store
      addDocumentPreview: (documentId: string, content: string[]) => {
        set({
          documentPreviews: {
            ...get().documentPreviews,
            [documentId]: content
          }
        });
      },

      getTable: (id = get().activeTableId) => {
        const current = get().tables.find(t => t.id === id);
        if (!current) {
          throw new Error(`No table with id ${id}`);
        }
        return current;
      },

      addTable: name => {
        const newTable = getBlankTable(name);
        set({
          tables: [...get().tables, newTable],
          activeTableId: newTable.id
        });
      },

      editTable: (id, table) => {
        set({ tables: where(get().tables, t => t.id === id, table) });
      },

      editActiveTable: table => {
        get().editTable(get().activeTableId, table);
      },

      switchTable: id => {
        if (get().tables.find(t => t.id === id)) {
          set({ activeTableId: id });
        }
      },

      deleteTable: async id => {
        const { tables, activeTableId, auth } = get();
        
        // Delete from the store
        const nextTables = tables.filter(t => t.id !== id);
        if (isEmpty(nextTables)) return;
        const nextActiveTable =
          activeTableId === id ? nextTables[0].id : activeTableId;
        set({ tables: nextTables, activeTableId: nextActiveTable });
        
        // Delete from the database if authenticated
        if (auth.isAuthenticated && auth.token) {
          try {
            // Import the deleteTableState function from the API
            const { deleteTableState } = await import("../../services/api/table-state");
            
            // Delete the table state from the database
            await deleteTableState(id);
          } catch (error) {
            console.error('Error deleting table state from database:', error);
            notifications.show({
              title: 'Delete failed',
              message: 'Failed to delete table state from database',
              color: 'red'
            });
          }
        }
      },

      insertColumnBefore: id => {
        const { getTable, editActiveTable } = get();
        editActiveTable({
          columns: insertBefore(
            getTable().columns,
            getBlankColumn(),
            id ? c => c.id === id : undefined
          )
        });
      },

      insertColumnAfter: id => {
        const { getTable, editActiveTable } = get();
        editActiveTable({
          columns: insertAfter(
            getTable().columns,
            getBlankColumn(),
            id ? c => c.id === id : undefined
          )
        });
      },

      editColumn: (id, column) => {
        // TODO (maybe): Handle column type changes
        const { getTable, editActiveTable } = get();
        editActiveTable({
          columns: where(getTable().columns, column => column.id === id, column)
        });
      },

      rerunColumns: ids => {
        const { getTable, rerunCells } = get();
        rerunCells(
          getTable()
            .rows.filter(row => !row.hidden)
            .flatMap(row => ids.map(id => ({ rowId: row.id, columnId: id })))
        );
      },

      clearColumns: ids => {
        const { getTable, editActiveTable } = get();
        editActiveTable({
          rows: getTable().rows.map(row => ({
            ...row,
            cells: omit(row.cells, ids)
          }))
        });
      },

      unwindColumn: id => {
        const { getTable, editActiveTable } = get();
        const { rows, columns } = getTable();
        const newRows: AnswerTableRow[] = [];
        const column = columns.find(c => c.id === id);
        if (!column || !isArrayType(column.type)) return;

        for (const row of rows) {
          const pivot = row.cells[id];
          if (!isArray(pivot)) continue;
          for (const part of pivot) {
            const newRow: AnswerTableRow = {
              id: cuid(),
              sourceData: row.sourceData,
              hidden: false,
              cells: {}
            };
            newRows.push(newRow);
            for (const column of columns) {
              newRow.cells[column.id] =
                column.id === id ? part : row.cells[column.id];
            }
          }
        }

        editActiveTable({
          rows: newRows,
          columns: where(columns, column => column.id === id, {
            type: toSingleType(column.type)
          })
        });
      },

      toggleAllColumns: hidden => {
        const { getTable, editActiveTable } = get();
        editActiveTable({
          columns: getTable().columns.map(column => ({ ...column, hidden }))
        });
      },

      deleteColumns: ids => {
        const { getTable, editActiveTable } = get();
        const table = getTable();
        editActiveTable({
          columns: table.columns
            .filter(column => !ids.includes(column.id))
            // Keep resolvedEntities for columns we're not deleting
            .map(col => ({ ...col })),
          rows: table.rows.map(row => ({
            ...row,
            cells: omit(row.cells, ids)
          })),
          globalRules: table.globalRules.map(rule => ({ 
            ...rule, 
            // Keep resolvedEntities for global rules
            resolvedEntities: rule.resolvedEntities || [] 
          }))
        });
      },

      reorderColumns: (sourceIndex: number, targetIndex: number) => {
        const { getTable, editActiveTable } = get();
        const columns = [...getTable().columns];
        
        // Don't do anything if the indices are the same
        if (sourceIndex === targetIndex) return;
        
        // Remove the column from the source index
        const [column] = columns.splice(sourceIndex, 1);
        
        // Insert the column at the target index
        columns.splice(targetIndex, 0, column);
        
        // Update the table
        editActiveTable({ columns });
      },

      insertRowBefore: id => {
        const { getTable, editActiveTable } = get();
        editActiveTable({
          rows: insertBefore(
            getTable().rows,
            getBlankRow(),
            id ? c => c.id === id : undefined
          )
        });
      },

      insertRowAfter: id => {
        const { getTable, editActiveTable } = get();
        editActiveTable({
          rows: insertAfter(
            getTable().rows,
            getBlankRow(),
            id ? c => c.id === id : undefined
          )
        });
      },

      // Updated fillRow to accept an options object
      fillRow: async (id: string, file: File, options = { showNotification: true }) => {
        const { activeTableId, editTable } = get();
        try {
          const document = await uploadFile(file);
          
          // Add document to tracked documents
          get().addDocument(document);
          
          // Start polling for status updates if document is processing
          if (document.status === 'processing') {
            get().pollDocumentStatus(document.id);
          }
          
          const sourceData: SourceData = {
            type: "document",
            document
          };
          
          // Update only the sourceData, preserving existing cells
          editTable(activeTableId, {
            rows: where(get().getTable(activeTableId).rows, r => r.id === id, row => ({
              ...row, // Keep existing properties
              sourceData, // Update the sourceData
              cells: {} // Reset cells
            }))
          });
          
          // Pass option to suppress the "Processing in progress" notification from rerunCells
          get().rerunRows([id], { suppressInProgressNotification: true });
          
          // Conditionally show notification based on options
          if (options.showNotification) {
            notifications.show({
              title: 'Document uploaded',
              message: `Successfully uploaded ${document.name}${document.status === 'processing' ? ' (processing in background)' : ''}`,
              color: 'green'
            });
          }
          
          return document;
        } catch (error) {
          console.error('Error uploading document:', error);
          notifications.show({
            title: 'Upload failed',
            message: error instanceof Error ? error.message : 'Unknown error',
            color: 'red'
          });
          return null;
        }
      },

      // Fill multiple rows, each with a file
      fillRows: async (files: File[]) => {
        const { activeTableId, editTable } = get();
        
        if (!activeTableId || !files.length) return;
        
        let successCount = 0;
        let errorCount = 0;
        const { rows } = get().getTable(activeTableId);
        const rowIds: string[] = [];
        const placeholderRows: AnswerTableRow[] = [];
        
        try {
          // Find the first available empty row
          let firstEmptyIndex = rows.findIndex(r => !r.sourceData);
          
          // Create rows for each file
          for (let i = 0; i < files.length; i++) {
            let id;
            // If we have an existing empty row, use it
            if (firstEmptyIndex !== -1 && i === 0) {
              id = rows[firstEmptyIndex].id;
              firstEmptyIndex = -1; // Mark as used
            } 
            // Otherwise create a new row
            else {
              id = cuid();
              const newRow = {
                ...getBlankRow(),
                id
              };
              placeholderRows.push(newRow);
            }
            
            rowIds.push(id);
          }
          
          // Add all new placeholder rows at once
          if (placeholderRows.length > 0) {
            editTable(activeTableId, { 
              rows: [...rows, ...placeholderRows] 
            });
          }
          
          // Process files in parallel
          const uploadPromises = files.map(async (file, index) => {
            const rowId = rowIds[index];
            
            try {
              // Upload the file
              const document = await uploadFile(file);
              
              // Add document to tracked documents
              get().addDocument(document);
              
              // Start polling for status updates if document is processing
              if (document.status === 'processing') {
                get().pollDocumentStatus(document.id);
              }
              
              // Update the row with actual document data
              const sourceData: SourceData = {
                type: "document",
                document
              };
              
              editTable(activeTableId, {
                rows: where(get().getTable(activeTableId).rows, r => r.id === rowId, row => ({
                  ...row, // Keep existing properties
                  sourceData, // Update the sourceData
                  cells: {} // Reset cells
                }))
              });
              
              // Run queries for this row
              get().rerunRows([rowId]);
              
              successCount++;
              return { success: true, file, document };
            } catch (error) {
              console.error(`Error uploading file ${file.name}:`, error);
              errorCount++;
              return { success: false, file, error };
            }
          });
          
          await Promise.all(uploadPromises);
          
          // Show notification about batch result
          const s = successCount > 1 ? 's' : '';
          notifications.show({
            title: 'Upload complete',
            message: `Successfully uploaded ${successCount} document${s}${errorCount > 0 ? `, ${errorCount} failed` : ''}`,
            color: 'green'
          });
          
        } catch (error) {
          console.error('Error in batch upload:', error);
          notifications.show({
            title: 'Batch upload error',
            message: error instanceof Error ? error.message : 'Unknown error',
            color: 'red'
          });
        }
      },

      // --- NEW ACTION: Ingest a single document by ID ---
      // Add types to parameters
      ingestSingleDocumentById: async (documentId: string, rowId: string, originalColumnId: string) => {
        const { auth, editTable, fillRow, editCells } = get();
        
        if (!auth.token) {
          console.error('Authentication token not found.');
          notifications.show({ title: 'Authentication Error', message: 'Please log in again.', color: 'red' });
          return { success: false };
        }
        
        try {
          console.log(`Processing document ID: ${documentId}`);

          // 1. Fetch text content first
          const textResponse = await fetch(`${API_ENDPOINTS.DOCUMENT_GET_TEXT(documentId)}`, {
            method: 'GET',
            credentials: 'include',
            headers: {
              'Accept': 'application/json',
              'Authorization': `Bearer ${auth.token}`,
            }
          });

          if (!textResponse.ok) {
            const errorText = await textResponse.text();
            console.error(`Failed to fetch text for document ID: ${documentId}`, textResponse.status, errorText);
            notifications.show({ title: 'Fetch Error', message: `Failed to get text for ${documentId}: ${textResponse.statusText}`, color: 'red' });
            return { success: false };
          }

          const textData = await textResponse.json();
          if (!textData.text) {
            console.error(`No text content for document ID: ${documentId}`);
            notifications.show({ title: 'Fetch Error', message: `No text content found for ${documentId}`, color: 'orange' });
            return { success: false };
          }

          // 2. Fetch metadata *only* to get the display name
          let displayName = `${documentId}.txt`; // Default display name
          try {
            const metadataResponse = await fetch(`${API_ENDPOINTS.DOCUMENT_GET_METADATA(documentId)}`, {
              method: 'GET',
              credentials: 'include',
              headers: {
                'Accept': 'application/json',
                'Authorization': `Bearer ${auth.token}`,
              }
            });

            if (metadataResponse.ok) {
              const metadataData = await metadataResponse.json();
              if (metadataData.metadata && metadataData.metadata.name) {
                displayName = metadataData.metadata.name; // Use real name for display later
              }
            } else {
               console.warn(`Could not fetch metadata for ${documentId}, using default name.`);
            }
          } catch (metadataError) {
            console.error(`Error fetching metadata for document ID: ${documentId}`, metadataError);
            // If metadata fetch fails, we'll use the default display name
          }

          // 3. Create the File object ALWAYS using .txt extension and text/plain type
          const uploadFileName = `${documentId}.txt`; // Ensure backend gets .txt
          const textBlob = new Blob([textData.text], { type: 'text/plain' });
          const file = new File([textBlob], uploadFileName, { type: 'text/plain' });

          // Save the original cell value before it gets cleared by fillRow (if applicable)
          const originalCellValue = get().getTable().rows.find(r => r.id === rowId)?.cells[originalColumnId];

          // 4. Upload the text file using fillRow, suppressing the individual notification
          await fillRow(rowId, file, { showNotification: false });

          // 5. Update the display name in the store AFTER successful upload
          const currentTable = get().getTable();
          editTable(currentTable.id, {
            rows: currentTable.rows.map(row => {
              if (row.id === rowId && row.sourceData?.type === 'document') {
                return {
                  ...row,
                  sourceData: {
                    ...row.sourceData,
                    document: {
                      ...row.sourceData.document,
                      name: displayName // Update the name here
                    }
                  }
                };
              }
              return row;
            })
          });

          // 6. Restore the original cell value in the document ID column if it existed
          if (originalColumnId && originalCellValue !== undefined) {
            editCells([{
              rowId: rowId,
              columnId: originalColumnId,
              cell: originalCellValue
            }]);
          }
          
          console.log(`Successfully processed and ingested document ID: ${documentId}`);
          return { success: true };

        } catch (error) {
          console.error(`Error processing document ID: ${documentId}`, error);
          notifications.show({ title: 'Ingestion Error', message: `Failed to process ${documentId}: ${error instanceof Error ? error.message : 'Unknown error'}`, color: 'red' });
          return { success: false };
        }
      },
      // --- END NEW ACTION ---

      // Updated to accept options
      rerunRows: (ids, options = { suppressInProgressNotification: false }) => {
        const { getTable, rerunCells } = get();
        rerunCells(
          getTable()
            .columns.filter(column => !column.hidden)
            .flatMap(column =>
              ids.map(id => ({ rowId: id, columnId: column.id }))
            ),
          options // Pass options down to rerunCells
        );
      },

      clearRows: ids => {
        const { getTable, editActiveTable } = get();
        const idSet = new Set(ids);
        editActiveTable({
          rows: where(getTable().rows, r => idSet.has(r.id), {
            sourceData: null,
            cells: {}
          })
        });
      },

      deleteRows: ids => {
        const { getTable, editActiveTable } = get();
        editActiveTable({
          rows: getTable().rows.filter(r => !ids.includes(r.id))
        });
      },

      editCells: (cells, tableId = get().activeTableId) => {
        const { getTable, editTable, saveTableState } = get();
        const valuesByRow = mapValues(
          groupBy(cells, c => c.rowId),
          c => c.map(c => [c.columnId, c.cell])
        );
        
        // Make the edits to the cells
        editTable(tableId, {
          rows: where(
            getTable(tableId).rows,
            r => valuesByRow[r.id],
            r => ({ cells: { ...r.cells, ...fromPairs(valuesByRow[r.id]) } })
          )
        });
        
        // Explicitly trigger a save after cell edits
        // This ensures changes are not lost, especially for manual edits
        console.log('Cell edit detected, triggering immediate table state save');
        saveTableState().catch(error => {
          console.error('Failed to save after cell edit:', error);
        });
      },

      // Updated to accept options
      rerunCells: (cells, options = { suppressInProgressNotification: false }) => {
        const { activeTableId, getTable, editTable, editCells } = get();
        const currentTable = getTable();
        const { columns, rows, globalRules, loadingCells } = currentTable;
        const colMap = keyBy(columns, c => c.id);
        const rowMap = keyBy(rows, r => r.id);
      
        // Check if there's already a batch query in progress
        const requestProgress = currentTable.requestProgress;
        if (requestProgress && requestProgress.inProgress) {
          // Conditionally show notification based on options
          if (!options.suppressInProgressNotification) {
            notifications.show({
              title: 'Processing in progress',
              message: 'Please wait for the current operation to complete before starting a new one.',
              color: 'blue',
              autoClose: 3000
            });
          }
          return; // Exit early
        }
      
        // Get the set of column IDs being rerun
        const rerunColumnIds = new Set(cells.map(cell => cell.columnId));
        
        // Get the set of row IDs being rerun
        const rerunRowIds = new Set(cells.map(cell => cell.rowId));
      
        // Create a Set of cell keys being rerun for easy lookup
        const rerunCellKeys = new Set(
          cells.map(cell => getCellKey(cell.rowId, cell.columnId))
        );
      
        // Don't clear resolved entities if we're processing new rows
        const isNewRow = cells.some(cell => {
          const row = rowMap[cell.rowId];
          return row && Object.keys(row.cells).length === 0;
        });
      
        if (!isNewRow) {
          editTable(activeTableId, {
            columns: columns.map(col => ({
              ...col,
              resolvedEntities: (col.resolvedEntities || []).filter(entity => {
                if (entity.source.type === 'column') {
                  const cellKey = getCellKey(
                    cells.find(cell => cell.columnId === entity.source.id)?.rowId || '',
                    entity.source.id
                  );
                  return !rerunCellKeys.has(cellKey);
                }
                return true;
              })
            })),
            globalRules: globalRules.map(rule => ({ 
              ...rule,
              resolvedEntities: (rule.resolvedEntities || []).filter(entity => {
                if (entity.source.type === 'global') {
                  const affectedRows = cells.filter(cell => 
                    rerunColumnIds.has(cell.columnId)
                  ).map(cell => cell.rowId);
                  return !affectedRows.some(rowId => rerunRowIds.has(rowId));
                }
                return true;
              })
            }))
          });
        }
      
        const batch = compact(
          cells.map(({ rowId, columnId }) => {
            const key = getCellKey(rowId, columnId);
            const column = colMap[columnId];
            const row = rowMap[rowId];
            return column &&
              row &&
              column.entityType.trim() &&
              column.generate &&
              !loadingCells[key]
              ? { key, column, row }
              : null;
          })
        );

        // If no valid cells to process, return early
        if (batch.length === 0) return;

        // Mark all cells as loading
        editTable(activeTableId, {
          loadingCells: {
            ...loadingCells,
            ...fromPairs(batch.map(m => [m.key, true]))
          }
        });

        // Prepare batch queries
        const batchQueries = batch.map(({ row, column: column_, key }) => {
          const column = cloneDeep(column_);
          let shouldRunQuery = true;
          let hasMatches = false;

          // Replace all column references with the row's answer to that column
          for (const [match, columnId] of column.query.matchAll(
            /@\[[^\]]+\]\(([^)]+)\)/g
          )) {
            hasMatches = true;
            const targetColumn = columns.find(c => c.id === columnId);
            if (!targetColumn) continue;
            const cell = row.cells[targetColumn.id];
            if (isNil(cell) || (isNil(cell) && isNil(row.sourceData))) {
              shouldRunQuery = false;
              break;
            }
            column.query = column.query.replace(match, String(cell));
          }
          
          if (!hasMatches && isNil(row.sourceData)) {
            shouldRunQuery = false;
          }
          
          return { 
            row, 
            column, 
            shouldRunQuery,
            globalRules,
            key
          };
        });

        // Filter out queries that shouldn't run
        const queriesToRun = batchQueries.filter(q => q.shouldRunQuery);
        
        // For queries that shouldn't run, clear their loading state immediately
        const skipKeys = batch
          .filter((_, i) => !batchQueries[i].shouldRunQuery)
          .map(({ key }) => key);
        
        if (skipKeys.length > 0) {
          editTable(activeTableId, {
            loadingCells: omit(getTable(activeTableId).loadingCells, skipKeys)
          });
        }
        
        // If no queries to run, return early
        if (queriesToRun.length === 0) return;
        
        // Helper to check if an entity matches any global rule patterns
        const isGlobalEntity = (entity: { original: string | string[] }) => {
          const originalText = Array.isArray(entity.original) 
            ? entity.original.join(' ') 
            : entity.original;
            
          return globalRules.some(rule => 
            rule.type === 'resolve_entity' && 
            rule.options?.some(pattern => 
              originalText.toLowerCase().includes(pattern.toLowerCase())
            )
          );
        };
        
        // Define a type for the entity parameter
        type EntityLike = { 
          original: string | string[]; 
          resolved: string | string[]; 
          source?: { type: string; id: string }; 
          entityType?: string 
        };
        
        // Process each response as it comes in
        const processQueryResult = (response: any, queryInfo: typeof queriesToRun[0]) => {
          if (!response) {
            console.error('Received null or undefined response for query:', queryInfo);
            return;
          }
          
          const { row, column, key } = queryInfo;
          const { answer, chunks, resolvedEntities } = response;
          
          if (!answer) {
            console.error('Response missing answer property:', response);
            return;
          }
          
          // Update cell value
          editCells(
            [{ rowId: row.id, columnId: column.id, cell: answer.answer }],
            activeTableId
          );
          
          // Get current state
          const currentTable = getTable(activeTableId);
          
          // Update table state with chunks and resolved entities
          editTable(activeTableId, {
            chunks: { ...currentTable.chunks, [key]: chunks || [] },
            loadingCells: omit(currentTable.loadingCells, key),
            columns: currentTable.columns.map(col => ({
              ...col,
              resolvedEntities: col.id === column.id 
                ? [
                    ...(col.resolvedEntities || []),
                    ...((resolvedEntities || [])
                      .filter((entity: EntityLike) => entity && !isGlobalEntity(entity))
                      .map((entity: EntityLike) => ({
                        ...entity,
                        entityType: column.entityType,
                        source: {
                          type: 'column' as const,
                          id: column.id
                        }
                      })) as ResolvedEntity[])
                  ]
                : (col.resolvedEntities || [])
            })),
            globalRules: currentTable.globalRules.map(rule => ({
              ...rule,
              resolvedEntities: rule.type === 'resolve_entity'
                ? [
                    ...(rule.resolvedEntities || []),
                    ...((resolvedEntities || [])
                      .filter((entity: EntityLike) => entity && isGlobalEntity(entity))
                      .map((entity: EntityLike) => ({
                        ...entity,
                        entityType: 'global',
                        source: {
                          type: 'global' as const,
                          id: rule.id
                        }
                      })) as ResolvedEntity[])
                  ]
                : (rule.resolvedEntities || [])
            }))
          });
        };
        
        // Handle errors for a specific query
        const handleQueryError = (error: any, queryInfo: typeof queriesToRun[0]) => {
          console.error(`Error running query for ${queryInfo.row.id}-${queryInfo.column.id}:`, error);
          
          // Clear loading state for this cell
          editTable(activeTableId, {
            loadingCells: omit(getTable(activeTableId).loadingCells, queryInfo.key)
          });
          
          // Just log the error to console, don't show notifications to user
          console.error('Query failed:', error instanceof Error ? error.message : 'Unknown error');
        };
        try {
          // Get current progress state for tracking
          const tableWithProgress = getTable(activeTableId);
          const existingProgress = tableWithProgress.requestProgress || { total: 0, completed: 0, inProgress: false };
          
          // IMPORTANT FIX: Always reset completed to 0 when starting a new batch
          // This ensures the progress bar starts from 0% and not 100%
          editTable(activeTableId, {
            requestProgress: {
              total: existingProgress.inProgress 
                ? existingProgress.total + queriesToRun.length 
                : queriesToRun.length,
              completed: 0, // Always reset to 0 when starting a new batch
              inProgress: true,
              error: false // Reset error state when starting new requests
            }
          });
        } catch (error) {
          console.error('Error updating progress state:', error);
          // Continue with the operation even if progress tracking fails
        }

        // Show a notification for very large batches
        if (queriesToRun.length > 50) {
          notifications.show({
            title: 'Processing requests',
            message: `Processing ${queriesToRun.length} cells in parallel.`,
            color: 'blue',
            autoClose: 3000
          });
        }
        // Verify we have queries to run
        if (queriesToRun.length === 0) {
          console.warn("No cells to rerun, aborting batch query operation");
          return;
        }
        
        // Determine batch size based on number of queries
        // With backend fix in place, we can use larger batch sizes
        const useBatchSize = queriesToRun.length > 50 ? 20 : 
                            queriesToRun.length > 20 ? 10 : 
                            queriesToRun.length > 10 ? 5 : 2;
        
        // Run batch queries with progressive updates
        runBatchQueries(
          queriesToRun,
          {
            batchSize: useBatchSize,
            maxRetries: 3,
            retryDelay: 1000,
            onQueryProgress: (result, index, total) => {
              // Process the result
              if (result && !result.error && index < queriesToRun.length) {
                processQueryResult(result, queriesToRun[index]);
              } else if (result && result.error && index < queriesToRun.length) {
                handleQueryError(result.error, queriesToRun[index]);
              }
              
              // Get the current progress state
              const currentTable = getTable(activeTableId);
              const currentProgress = currentTable.requestProgress || { 
                total: queriesToRun.length, 
                completed: 0, 
                inProgress: true 
              };
              
              // Calculate the new completed count - use the total from the API module
              // which is tracking the exact number of processed queries
              const newCompletedCount = typeof total === 'number' ? 
                Math.min(total, currentProgress.total) : // Use the total from API if it's a number
                currentProgress.completed + 1;           // Fallback to incrementing by 1
              
              // Only update if the count has changed
              if (newCompletedCount !== currentProgress.completed) {
                // Log for debugging
                // console.log(`Store updating progress: ${newCompletedCount}/${currentProgress.total}`);
                
                // Update the progress state
                editTable(activeTableId, {
                  requestProgress: {
                    ...currentProgress,
                    completed: newCompletedCount
                  }
                });
              }
            },
            onBatchProgress: (results, batchIndex, totalBatches) => {
              console.log(`Processed batch ${batchIndex + 1}/${totalBatches}`);
              
          // Process each result in the batch
          // Validate useBatchSize is at least 1 to prevent division by zero
          const safeBatchSize = Math.max(1, useBatchSize);
          const batchStart = batchIndex * safeBatchSize;
          
          // Validate results is an array and has valid length
          if (!Array.isArray(results)) {
            console.error('Batch results is not an array:', results);
            return;
          }
          
          // Make sure we have results to process
          if (results.length === 0) {
            console.warn('Empty batch results array received');
            return;
          }
          
          // Safely process each result
          for (let resultIndex = 0; resultIndex < results.length; resultIndex++) {
            const queryIndex = batchStart + resultIndex;
            if (queryIndex < queriesToRun.length) {
              try {
                // Check that the result is valid before processing
                if (results[resultIndex] !== undefined && results[resultIndex] !== null) {
                  processQueryResult(results[resultIndex], queriesToRun[queryIndex]);
                } else {
                  console.warn(`Undefined or null result at index ${resultIndex}, skipping`);
                }
              } catch (error) {
                console.error(`Error processing result at index ${resultIndex}:`, error);
              }
            }
          }
              
              // Update progress counter for the batch with strict validation
              try {
                const currentTable = getTable(activeTableId);
                if (!currentTable) {
                  console.warn('Table not found when updating progress counter');
                  return;
                }
                
                const currentProgress = currentTable.requestProgress || { 
                  total: queriesToRun.length, 
                  completed: 0, 
                  inProgress: true 
                };
                
                // Make sure results.length is a valid number
                const validCompletedResults = Array.isArray(results) ? results.length : 0;
                
                // Ensure we never exceed the total count
                const completedCount = Math.min(
                  currentProgress.completed + validCompletedResults,
                  currentProgress.total
                );
                
                // Log progress for debugging
                console.log(`Progress update: ${completedCount}/${currentProgress.total} (${Math.round(completedCount/currentProgress.total*100)}%)`);
                
                // Only update if the count is valid and doesn't exceed total
                if (completedCount <= currentProgress.total) {
                  editTable(activeTableId, {
                    requestProgress: {
                      ...currentProgress,
                      completed: completedCount
                    }
                  });
                } else {
                  console.error(`Invalid progress count: ${completedCount}/${currentProgress.total}`);
                }
              } catch (error) {
                console.error('Error updating batch progress:', error);
                // Continue processing even if progress tracking fails
              }
            }
          }
        ).then(() => {
          try {
            // Mark progress as complete when all requests are done
            const currentTable = getTable(activeTableId);
            const currentProgress = currentTable.requestProgress || { total: 0, completed: 0, inProgress: false };
            
            // Ensure completed count never exceeds total
            const finalCompletedCount = Math.min(currentProgress.completed, currentProgress.total);
            
            // Log final progress for debugging
            console.log(`Final progress: ${finalCompletedCount}/${currentProgress.total} (${Math.round(finalCompletedCount/currentProgress.total*100)}%)`);
            
            editTable(activeTableId, {
              requestProgress: {
                total: currentProgress.total,
                completed: finalCompletedCount,
                inProgress: false
              }
            });
          } catch (error) {
            console.error('Error updating progress state on completion:', error);
            // Continue even if progress tracking fails
          }
          
          // Notification is now handled by the KtProgressBar component
        }).catch((error: any) => {
          console.error('Error running batch queries:', error);
          
          // Get access to saveTableState function
          const { saveTableState } = get();
          
          // Clear loading state for all remaining cells in the batch that haven't been processed
          const remainingLoadingCells = getTable(activeTableId).loadingCells;
          if (Object.keys(remainingLoadingCells).length > 0) {
            editTable(activeTableId, {
              loadingCells: {}
            });
          }
          
          try {
            // Get the current table and progress state
            const currentTable = getTable(activeTableId);
            const currentProgress = currentTable.requestProgress || { 
              total: queriesToRun.length, 
              completed: 0, 
              inProgress: true 
            };
            
            // Mark progress as complete but with error, preserving any successful completions
            editTable(activeTableId, {
              requestProgress: {
                total: queriesToRun.length,
                completed: currentProgress.completed, // Keep track of any successful completions
                inProgress: false,
                error: true
              }
            });
            
            // Show minimal notification about the failure
            notifications.show({
              title: 'Some queries failed',
              message: `${currentProgress.completed} of ${queriesToRun.length} queries processed successfully`,
              color: 'orange',
              autoClose: 3000
            });
            
            // Save current state to prevent data loss
            if (saveTableState) {
              saveTableState().catch(saveError => {
                console.error('Failed to save after batch error:', saveError);
              });
            }
          } catch (processError) {
            console.error('Error updating progress state on error:', processError);
            // Continue even if progress tracking fails
          }
          
          // Log detailed error information to console
          console.error('Batch query failed:', error instanceof Error ? error.message : 'Unknown error');
        });
      },

      clearCells: cells => {
        const { getTable, editActiveTable, saveTableState } = get();
        const columnsByRow = mapValues(
          groupBy(cells, c => c.rowId),
          c => c.map(c => c.columnId)
        );
        
        // Clear the cells
        editActiveTable({
          rows: where(
            getTable().rows,
            r => columnsByRow[r.id],
            r => ({ cells: omit(r.cells, columnsByRow[r.id]) })
          )
        });
        
        // Explicitly save after clearing cells
        console.log('Cells cleared, triggering immediate table state save');
        saveTableState().catch(error => {
          console.error('Failed to save after clearing cells:', error);
        });
      },

      addGlobalRules: rules => {
        const { getTable, editActiveTable } = get();
        editActiveTable({
          globalRules: [
            ...getTable().globalRules,
            ...rules.map(rule => ({ id: cuid(), ...rule }))
          ]
        });
      },

      editGlobalRule: (id, rule) => {
        const { getTable, editActiveTable } = get();
        editActiveTable({
          globalRules: where(
            getTable().globalRules,
            rule => rule.id === id,
            rule
          )
        });
      },

      deleteGlobalRules: ids => {
        const { getTable, editActiveTable } = get();
        editActiveTable({
          globalRules: !ids
            ? []
            : getTable().globalRules.filter(rule => !ids.includes(rule.id))
        });
      },

      openChunks: cells => {
        get().editActiveTable({
          openedChunks: cells.map(c => getCellKey(c.rowId, c.columnId))
        });
      },

      closeChunks: () => {
        get().editActiveTable({ openedChunks: [] });
      },

      addFilter: filter => {
        const { getTable, editActiveTable } = get();
        editActiveTable({
          filters: [...getTable().filters, { id: cuid(), ...filter }]
        });
        get().applyFilters();
      },

      editFilter: (id, filter) => {
        const { getTable, editActiveTable } = get();
        editActiveTable({
          filters: where(getTable().filters, filter => filter.id === id, filter)
        });
        get().applyFilters();
      },

      deleteFilters: ids => {
        const { getTable, editActiveTable } = get();
        editActiveTable({
          filters: !ids
            ? []
            : getTable().filters.filter(filter => !ids.includes(filter.id))
        });
        get().applyFilters();
      },

      applyFilters: () => {
        const { getTable, editActiveTable } = get();
        const { rows, columns, filters } = getTable();
        editActiveTable({
          rows: rows.map(row => {
            const visible = filters.every(filter => {
              if (!filter.value.trim()) return true;
              const column = columns.find(
                column => column.id === filter.columnId
              );
              if (!column) return true;
              const cell = row.cells[column.id];
              if (isNil(cell)) return true;
              const contains = castArray(cell)
                .map(value => String(value).toLowerCase())
                .some(value =>
                  value.includes(filter.value.trim().toLowerCase())
                );
              return filter.criteria === "contains" ? contains : !contains;
            });
            return { ...row, hidden: !visible };
          })
        });
      },

      // Debounce timer for saveTableState
      _saveTableStateTimer: null as ReturnType<typeof setTimeout> | null,
      
      // Automatic persistence methods
      saveTableState: async () => {
        const { activeTableId, getTable, auth, _saveTableStateTimer } = get();
        
        // Only save if authenticated
        if (!auth.isAuthenticated || !auth.token) {
          console.log('Not saving table state: User not authenticated');
          return Promise.resolve();
        }
        
        // Clear any existing timer
        if (_saveTableStateTimer) {
          clearTimeout(_saveTableStateTimer);
        }
        
        // Set a new timer with shorter delay (500ms instead of 2000ms)
        const timer = setTimeout(async () => {
          console.log('Save timer triggered, preparing table state for saving...');
          try {
            const table = getTable(activeTableId);
            if (!table) {
              console.error('No active table found, cannot save');
              return;
            }
            
            // Debug info
            console.log(`Saving table: ${table.id} with ${table.rows.length} rows and ${table.columns.length} columns`);
            // console.log(`Row cell counts: ${table.rows.slice(0, 3).map(r => Object.keys(r.cells).length).join(', ')}...`);
            
            // Check for large data payload
            const estimatedSize = JSON.stringify(table).length;
            const megabytes = estimatedSize / (1024 * 1024);
            // console.log(`Estimated payload size: ${megabytes.toFixed(2)} MB`);
            
            // Handle large tables by optimizing the saved data
            let tableToSave = table;
            if (table.rows.length > 1000 || megabytes > 10) {
              console.log('Large table detected, optimizing data for storage');
              
              // Create optimized version with only essential data
              // Start with a clean object and only copy what we need
              tableToSave = {
                // Core properties that must be preserved
                id: table.id,
                name: table.name,
                columns: table.columns,
                // Only include essential fields for rows to reduce size
                rows: table.rows.map(row => ({
                  id: row.id,
                  hidden: row.hidden || false,
                  cells: row.cells,
                  sourceData: row.sourceData,
                })),
                globalRules: table.globalRules,
                filters: table.filters,
                // Required properties with minimal data
                chunks: {},
                openedChunks: [],
                loadingCells: {},
                uploadingFiles: false
              };
              
              console.log('Table optimized for storage, removed temporary data');
            }
            
            try {
              // Try direct update first (most common case)
              try {
                // console.log(`Attempting direct update for table ${table.id}`);
                await apiUpdateTableState(table.id, tableToSave);
                // console.log('Table state update successful via direct PUT');
                return;
              } catch (updateError) {
                // If update fails, provide detailed error
                console.warn(`Update error: ${updateError instanceof Error ? updateError.message : 'Unknown error'}`);
                
                // If it's a 404, create a new table state
                if (updateError instanceof Error && updateError.message.includes('not found')) {
                  console.log(`Table ${table.id} not found, creating new table state`);
                  await apiSaveTableState(table.id, table.name, tableToSave);
                  console.log('Table state created successfully via POST');
                  return;
                }
                
                // For other errors, use the fallback with explicit existence check
                console.log('Direct update failed, checking if table exists...');
                
                // Fallback: check if the table state exists by listing all table states
                const response = await listTableStates();
                const existingState = response.items?.find(item => item.id === table.id);
                
                if (existingState) {
                  // If it exists, update it
                  console.log(`Table ${table.id} found in list, updating`);
                  await apiUpdateTableState(table.id, tableToSave);
                  console.log('Table state updated successfully via fallback PUT');
                } else {
                  // If it doesn't exist, create a new one
                  console.log(`Table ${table.id} not found in list, creating`);
                  await apiSaveTableState(table.id, table.name, tableToSave);
                  console.log('Table state created successfully via fallback POST');
                }
              }
            } catch (apiError) {
              // Log the error with more details
              console.error('API error saving table state:', apiError);
              console.error('Error details:', apiError instanceof Error ? apiError.message : 'Unknown error');
            }
          } catch (error) {
            console.error('Error preparing table state for save:', error);
            console.error('Stack trace:', error instanceof Error ? error.stack : 'No stack trace');
          } finally {
            // Clear the timer reference
            set({ _saveTableStateTimer: null });
          }
        }, 500); // Reduced from 2000ms to 500ms for more responsive saving
        
        // Store the timer
        set({ _saveTableStateTimer: timer });
        
        return Promise.resolve();
      },
      
      loadLatestTableState: async () => {
        const { auth } = get();
        
        // Only load if authenticated
        if (!auth.isAuthenticated || !auth.token) {
          return Promise.resolve();
        }
        
        try {
          // Set loading state
          set({ isLoading: true });
          
          try {
            // Get all table states
            const response = await listTableStates();
            
            // If no table states, return
            if (!response.items || response.items.length === 0) {
              set({ isLoading: false });
              return Promise.resolve();
            }
            
            // Sort by updated_at to get the most recent first
            const sortedStates = response.items.sort((a, b) => 
              new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
            );
            
            // Load all table states into the store
            const tables = sortedStates.map(state => ({
              id: state.id,
              name: state.name,
              columns: state.data.columns || [],
              rows: state.data.rows || [],
              globalRules: state.data.globalRules || [],
              filters: state.data.filters || [],
              chunks: state.data.chunks || {},
              openedChunks: state.data.openedChunks || [],
              loadingCells: {},
              uploadingFiles: false
            }));
            
            // Set all tables and make the most recent one active
            set({
              tables,
              activeTableId: tables.length > 0 ? tables[0].id : get().activeTableId,
              isLoading: false
            });
            
            return Promise.resolve();
          } catch (apiError) {
            console.error('Error loading table state from API:', apiError);
            set({ isLoading: false });
            return Promise.resolve();
          }
        } catch (error) {
          console.error('Error processing loaded table state:', error);
          set({ isLoading: false });
          return Promise.resolve();
        }
      },
      // Import CSV data into the grid
      importCsvData: async (data, preserveExistingColumns = false) => {
        const { getTable, editActiveTable } = get();
        
        try {
          // If there's no data, return
          if (!data || data.length === 0) {
            return;
          }
          
          // Get current table
          const table = getTable();
          
          // Get header row (first row of CSV)
          const headerRow = data[0];
          const dataRows = data.slice(1);
          
          // Create new columns based on header row
          const newColumns = headerRow.map((header) => {
            // Check if a column with this name already exists
            const existingColumn = table.columns.find(col => 
              col.entityType.toLowerCase() === header.toLowerCase() ||
              col.query.toLowerCase() === header.toLowerCase()
            );
            
            if (existingColumn && preserveExistingColumns) {
              return existingColumn;
            }
            
            // Create a new column for each header
            return {
              ...getBlankColumn(),
              entityType: header || 'Column',
              query: header || 'Column', 
              generate: false, // Don't auto-generate for imported data
              type: "str" as const // Default to string type for CSV data
            };
          });
          
          // Create rows with cell data
          const newRows = dataRows.map(rowData => {
            const row = getBlankRow();
            
            // Add cell data
            const cells: Record<string, any> = {};
            rowData.forEach((cellValue, index) => {
              if (index < newColumns.length) {
                cells[newColumns[index].id] = cellValue ? cellValue : '';
              }
            });
            
            return {
              ...row,
              cells
            };
          });
          
          if (preserveExistingColumns) {
            // Merge new columns with existing ones
            const existingColumnIds = new Set(table.columns.map(col => col.id));
            const columnsToAdd = newColumns.filter(col => !existingColumnIds.has(col.id));
            
            editActiveTable({
              columns: [...table.columns, ...columnsToAdd],
              rows: [...table.rows, ...newRows]
            });
          } else {
            // Replace existing columns and rows with new ones from CSV
            editActiveTable({
              columns: newColumns,
              rows: newRows
            });
          }
          
          // Force a UI refresh by saving the state
          get().saveTableState();
          
          return Promise.resolve();
        } catch (error) {
          console.error('Error importing CSV data:', error);
          return Promise.reject(error);
        }
      },

      clear: allTables => {
        if (allTables) {
          // Clear all browser storage
          try {
            // Clear sessionStorage
            sessionStorage.clear();
            
            // Clear localStorage
            localStorage.clear();
            
            console.log('All browser storage cleared');
          } catch (e) {
            console.error('Error clearing browser storage:', e);
          }
          
          // Reset the store to initial state
          set({
            ...getInitialData(),
            documentPreviews: {} // Clear document previews when clearing all tables
          });
        } else {
          const { id, name, ...table } = getBlankTable();
          get().editActiveTable({
            ...table,
            columns: table.columns.map(col => ({ ...col, resolvedEntities: [] })),
            globalRules: table.globalRules.map(rule => ({ ...rule, resolvedEntities: [] }))
          });
        }
      },

      // Documents methods
      addDocument: (document) => {
        set({
          documents: {
            ...get().documents,
            [document.id]: document
          }
        });
      },

      updateDocumentStatus: (documentId, status) => {
        const { documents } = get();
        
        if (documents[documentId]) {
          // Make sure status is one of the valid enum values
          const validStatus = status === 'processing' || status === 'completed' || status === 'failed' 
            ? status 
            : 'completed'; // Default to completed for unknown status values
            
          set({
            documents: {
              ...documents,
              [documentId]: {
                ...documents[documentId],
                status: validStatus
              }
            }
          });
        }
      },

      checkDocumentStatus: async (documentId) => {
        try {
          const { documents } = get();
          
          // Only check status for processing documents
          if (!documents[documentId] || documents[documentId].status !== 'processing') {
            return;
          }
          
          const result = await checkDocumentStatus(documentId);
          
          if (result.status !== documents[documentId].status) {
            // Update document status if changed
            get().updateDocumentStatus(documentId, result.status);
            
            // ADDED: Also update the document status in table state
            const { activeTableId, getTable, editTable } = get();
            if (activeTableId) {
              const table = getTable(activeTableId);
              
              // Check if any rows need to be updated
              let rowsUpdated = false;
              
              // Create a copy of the rows array to modify
              const updatedRows = [...table.rows];
              
              // Find rows with this document and update their status
              updatedRows.forEach(row => {
                if (row.sourceData?.type === 'document' &&
                    row.sourceData.document?.id === documentId &&
                    row.sourceData.document.status !== result.status) {
                  
                  // Mark that we found a row to update
                  rowsUpdated = true;
                  
                  // TypeScript safe way to update the status
                  const docData = row.sourceData as {
                    type: 'document';
                    document: {
                      id: string;
                      status: "processing" | "completed" | "failed";
                      [key: string]: any;
                    }
                  };
                  
                  // Update the status
                  docData.document.status = result.status as "processing" | "completed" | "failed";
                }
              });
              
              if (rowsUpdated) {
                console.log(`Updating document ${documentId} status in table state from processing to ${result.status}`);
                
                // Update the table with the modified rows
                editTable(activeTableId, { rows: updatedRows });
                
                // Save the table state to persist the changes
                get().saveTableState().catch(error => {
                  console.error('Failed to save table state after updating document status:', error);
                });
                
                // Save table state
                get().saveTableState().catch(error => {
                  console.error('Failed to save table state after updating document status:', error);
                });
              }
            }
            
            // Show notification on completion or failure
            if (result.status === 'completed') {
              notifications.show({
                title: 'Document processed',
                message: `Document ${documents[documentId].name} has been processed successfully`,
                color: 'green'
              });
            } else if (result.status === 'failed') {
              notifications.show({
                title: 'Document processing failed',
                message: `Processing of document ${documents[documentId].name} has failed`,
                color: 'red'
              });
            }
          }
        } catch (error) {
          console.error(`Error checking status for document ${documentId}:`, error);
        }
      },

      pollDocumentStatus: async (documentId, initialInterval = 3000, maxAttempts = 20) => {
        let attempts = 0;
        
        const poll = async () => {
          await get().checkDocumentStatus(documentId);
          
          const { documents } = get();
          const document = documents[documentId];
          
          // Stop polling if document is complete, failed, or we've reached max attempts
          if (!document || document.status !== 'processing' || attempts >= maxAttempts) {
            return;
          }
          
          // Continue polling with exponential backoff
          attempts++;
          
          // Calculate next interval with exponential backoff
          // Double the interval each time, capped at 3 minutes
          const nextInterval = Math.min(
            initialInterval * Math.pow(2, attempts - 1),
            180000 // 3 minutes max
          );
          
          console.log(`Scheduling next document status check in ${nextInterval/1000} seconds (attempt ${attempts}/${maxAttempts})`);
          setTimeout(poll, nextInterval);
        };
        
        // Start polling
        poll();
      }
    }),
    {
      name: "store",
      version: 12, // Increment version due to schema change
      partialize: (state) => {
        // Store minimal data regardless of authentication state
        return {
          // Only store auth state and color scheme
          auth: state.auth,
          colorScheme: state.colorScheme
        };
      },
      // Add storage option to use sessionStorage instead of localStorage
      // This will clear when the browser is closed
      storage: {
        getItem: (name) => {
          try {
            const value = sessionStorage.getItem(name);
            return value ? JSON.parse(value) : null;
          } catch (e) {
            console.error('Error reading from sessionStorage:', e);
            return null;
          }
        },
        setItem: (name, value) => {
          try {
            sessionStorage.setItem(name, JSON.stringify(value));
          } catch (e) {
            console.error('Error writing to sessionStorage:', e);
          }
        },
        removeItem: (name) => {
          try {
            sessionStorage.removeItem(name);
          } catch (e) {
            console.error('Error removing from sessionStorage:', e);
          }
        }
      }
    }
  )
);

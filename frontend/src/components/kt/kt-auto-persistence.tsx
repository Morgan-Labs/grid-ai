import { useEffect, useRef } from "react";
import { useStore } from "@config/store";
import { debounce } from "lodash-es";

/**
 * Component that automatically saves and loads table state
 * This component doesn't render anything, it just adds the auto-save functionality
 */
export function KtAutoPersistence() {
  const isAuthenticated = useStore(state => state.auth.isAuthenticated);
  const tableState = useStore(state => {
    try {
      const table = state.getTable();
      return {
        id: table.id,
        name: table.name,
        columns: table.columns,
        rows: table.rows,
        globalRules: table.globalRules,
        filters: table.filters,
        chunks: table.chunks,
        openedChunks: table.openedChunks,
      };
    } catch (e) {
      return null;
    }
  });
  
  // Track if this is the first load
  const isFirstLoad = useRef(true);
  
  // Track the previous table state to detect changes
  const prevTableStateRef = useRef<any>(null);
  
  // Track the last save time
  const lastSaveTimeRef = useRef<number>(0);
  
  // Load the latest table state when the component mounts and the user is authenticated
  useEffect(() => {
    if (isAuthenticated) {
      // Load the latest table state
      useStore.getState().loadLatestTableState()
        .then(() => {
          // No logs or notifications for normal operation
          isFirstLoad.current = false;
          
          // Initialize the previous state reference after loading with the same format
          // as we'll use for change detection
          try {
            const table = useStore.getState().getTable();
            prevTableStateRef.current = {
              id: table.id,
              name: table.name,
              columnCount: table.columns.length,
              rowCount: table.rows.length,
              columnsHash: JSON.stringify(table.columns.map(c => ({ 
                id: c.id, 
                entityType: c.entityType,
                query: c.query,
                type: c.type,
                generate: c.generate,
                rules: c.rules
              }))),
              rowsHash: JSON.stringify(table.rows.map(r => ({
                id: r.id,
                hidden: r.hidden,
                sourceDataId: r.sourceData?.type === 'document' ? r.sourceData.document.id : null,
                cellCount: Object.keys(r.cells).length
              }))),
              globalRulesHash: JSON.stringify(table.globalRules),
              filtersHash: JSON.stringify(table.filters)
            };
          } catch (e) {
            prevTableStateRef.current = null;
          }
        })
        .catch((error) => {
          // Just log the error without showing notifications
          console.error('Error loading table state:', error);
        });
    }
  }, [isAuthenticated]);
  
  // Save the table state only when it actually changes
  useEffect(() => {
    if (!isAuthenticated || !tableState) return;
    
    // Skip the first render after loading
    if (isFirstLoad.current) return;
    
    // Check if there are actual changes by comparing with previous state
    // We include cell content in the check to detect cell content changes
    const currentState = {
      id: tableState.id,
      name: tableState.name,
      // Only include the essential properties that would trigger a save
      // Exclude large objects like chunks that change frequently but don't need to be saved
      columnCount: tableState.columns.length,
      rowCount: tableState.rows.length,
      // Use a hash of the data instead of the full objects
      columnsHash: JSON.stringify(tableState.columns.map(c => ({ 
        id: c.id, 
        entityType: c.entityType,
        query: c.query,
        type: c.type,
        generate: c.generate,
        rules: c.rules
      }))),
      // Include a sample of cell content to detect changes in cell data
      rowsHash: JSON.stringify(tableState.rows.length > 1000 ? 
        // For large tables (>1000 rows), use sampling for better performance
        (() => {
          console.log(`Large table detected (${tableState.rows.length} rows), using sampling for change detection`);
          // Sample rows: first 50 plus every 50th after that
          const sampledRows = [
            ...tableState.rows.slice(0, 50),
            ...tableState.rows.slice(50).filter((_, i) => i % 50 === 0)
          ];
          return sampledRows.map(r => ({
            id: r.id,
            hidden: r.hidden,
            sourceDataId: r.sourceData?.type === 'document' ? r.sourceData.document.id : null,
            // Sample just 3 cells per row for large tables
            cellSample: Object.keys(r.cells).slice(0, 3).map(key => ({ 
              key, value: r.cells[key] 
            })),
            cellCount: Object.keys(r.cells).length
          }));
        })() 
        : 
        // For smaller tables, check more cells
        tableState.rows.map(r => ({
          id: r.id,
          hidden: r.hidden,
          sourceDataId: r.sourceData?.type === 'document' ? r.sourceData.document.id : null,
          // Sample up to 10 cells per row for smaller tables
          cellSample: Object.keys(r.cells).slice(0, 10).map(key => ({ 
            key, value: r.cells[key] 
          })),
          cellCount: Object.keys(r.cells).length
        }))
      ),
      globalRulesHash: JSON.stringify(tableState.globalRules),
      filtersHash: JSON.stringify(tableState.filters)
    };
    
    // Check for changes - we'll do a more relaxed check to ensure more frequent saves
    const hasChanged = !prevTableStateRef.current || 
                      currentState.columnCount !== prevTableStateRef.current.columnCount ||
                      currentState.rowCount !== prevTableStateRef.current.rowCount ||
                      currentState.columnsHash !== prevTableStateRef.current.columnsHash ||
                      currentState.rowsHash !== prevTableStateRef.current.rowsHash ||
                      currentState.globalRulesHash !== prevTableStateRef.current.globalRulesHash ||
                      currentState.filtersHash !== prevTableStateRef.current.filtersHash;
    
    // Always update the previous state reference with a deep clone
    prevTableStateRef.current = { ...currentState };
    
    // If nothing changed, return early
    if (!hasChanged) {
      return;
    }
    
    // Reduce throttling time to more frequently save changes (3 seconds between saves)
    const currentTime = Date.now();
    if (currentTime - lastSaveTimeRef.current < 3000) {
      return;
    }
    
    // Debounce the save operation to avoid too many saves
    const debouncedSave = debounce(() => {
      // Update the last save time
      lastSaveTimeRef.current = Date.now();
      
      // Try to save to the backend with debug logging
      // console.log('Attempting to save table state after detecting changes');
      useStore.getState().saveTableState()
        .then(() => {
          console.log('Table state saved successfully (auto-save)');
        })
        .catch((error) => {
          // Log errors with more detail
          console.error('Error in auto-saving table state:', error);
        });
    }, 2000); // Reduced from 5 seconds to 2 seconds for more responsive saving
    
    debouncedSave();
    
    // Clean up the debounce function
    return () => {
      debouncedSave.cancel();
    };
  }, [isAuthenticated, tableState]);
  
  // This component doesn't render anything
  return null;
}

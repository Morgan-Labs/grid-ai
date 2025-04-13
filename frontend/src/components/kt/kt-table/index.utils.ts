import { ReactGridProps, HeaderCell, CellChange } from "@silevis/reactgrid";
// Remove unused mapValues, castArray, CellValue
import { isEmpty, uniqBy } from "lodash-es"; 
import { KtColumnCell, KtRowCell, KtCell } from "./kt-cells";
import { useStore } from "@config/store";
import { pack, plur } from "@utils/functions";
// import { CellValue } from "@config/store/store.types"; // Removed unused import
import { notifications } from "@utils/notifications";

export type Cell = HeaderCell | KtColumnCell | KtRowCell | KtCell;

export const HEADER_ROW_ID = "header-row";
export const SOURCE_COLUMN_ID = "source-column";

export const handleCellChange = (changes: CellChange[]) => {
  const filteredChanges = (changes as CellChange<Cell>[]).filter(
    change =>
      change.rowId !== HEADER_ROW_ID &&
      change.columnId !== SOURCE_COLUMN_ID &&
      change.previousCell.type === "kt-cell" &&
      change.newCell.type === "kt-cell"
  );
  useStore.getState().editCells(
    filteredChanges.map(change => ({
      rowId: String(change.rowId),
      columnId: String(change.columnId),
      cell: (change.newCell as KtCell).cell
    }))
  );
};

export const handleContextMenu: Required<ReactGridProps>["onContextMenu"] = (
  selectedRowIds_,
  selectedColIds_,
  _,
  options,
  selectedRanges
) => {
  const store = useStore.getState();
  const table = store.getTable(); // Get current table state
  const rowIds = selectedRowIds_
    .filter(rowId => rowId !== HEADER_ROW_ID)
    .map(String);
  const colIds = selectedColIds_
    .filter(colId => colId !== SOURCE_COLUMN_ID)
    .map(String);

  // Reverted variable name back to 'cells' for consistency
  const cells = uniqBy(
    selectedRanges
      .flat()
      .filter(c => c.rowId !== HEADER_ROW_ID && c.columnId !== SOURCE_COLUMN_ID),
    c => `${c.rowId}-${c.columnId}`
  ).map(cell => ({ // Keep original mapping structure
    rowId: String(cell.rowId),
    columnId: String(cell.columnId),
  }));


  return pack([
    // Options for single or multiple cell selections (but not full rows/columns)
    !isEmpty(cells) &&
      isEmpty(rowIds) &&
      isEmpty(colIds) && [
        // --- Keep existing cell options using original 'cells' variable ---
        {
          id: "rerun-cells",
          label: `Rerun ${plur("cell", cells.length)}`, // Use cells.length for plur
          handler: () => store.rerunCells(cells)
        },
        {
          id: "clear-cells",
          label: `Clear ${plur("cell", cells.length)}`, // Use cells.length for plur
          handler: () => store.clearCells(cells)
        },
        {
          id: "chunks",
          label: "View chunks",
          handler: () => store.openChunks(cells)
        },
        // --- NEW: Ingest SFID Option for Cells (Added here) ---
        {
          id: "ingest-sfid-cells",
          // Revert plur cast, use original cells.length
          label: `Ingest ${plur("Document", cells.length)}`, 
          handler: async () => {
            // Get full cell details only when the handler is invoked
            const cellsToProcess = cells.map(loc => {
              const row = table.rows.find(r => r.id === loc.rowId);
              const cellValue = row?.cells[loc.columnId]; // Use original columnId
              let textValue: string | null = null;
              // Extract text value regardless of cell type
              if (typeof cellValue === 'string') {
                textValue = cellValue;
              } else if (typeof cellValue === 'object' && cellValue && 'text' in cellValue) {
                // Ensure 'text' property exists and is a string before accessing
                textValue = typeof cellValue.text === 'string' ? cellValue.text : null;
              }
              return { ...loc, textValue: textValue?.trim() };
            }).filter(c => c.textValue); // Filter for cells with actual text *inside* the handler

            if (isEmpty(cellsToProcess)) {
              notifications.show({ title: 'No IDs Found', message: 'Selected cells are empty or do not contain text.', color: 'orange' });
              return;
            }

            const totalToIngest = cellsToProcess.length;
            
            // Set initial progress state in the store
            store.editActiveTable({
              requestProgress: {
                total: totalToIngest,
                completed: 0,
                inProgress: true,
                error: false
              }
            });

            let successCount = 0;
            // Process sequentially
            for (const cell of cellsToProcess) {
              // Check textValue again just in case
              if (cell.textValue) {
                 const result = await store.ingestSingleDocumentById(
                   cell.textValue,
                   cell.rowId,
                   cell.columnId // Pass the original column ID
                 );
                 if (result.success) {
                   successCount++;
                 }
                 // Update progress after each attempt
                 const currentProgress = store.getTable().requestProgress || { total: totalToIngest, completed: 0, inProgress: true };
                 store.editActiveTable({
                   requestProgress: {
                     ...currentProgress,
                     completed: Math.min(currentProgress.completed + 1, totalToIngest) 
                   }
                 });
              }
            }

            // Finalize progress state in the store
            store.editActiveTable({
              requestProgress: {
                total: totalToIngest,
                completed: totalToIngest, // Mark as fully completed for UI
                inProgress: false,
                error: successCount < totalToIngest // Set error flag if not all succeeded
              }
            });
            
            // No separate notification needed here, progress bar handles it.
          }
        }
        // --- End Ingest SFID Option for Cells ---
      ],
    // --- Keep all other existing menu groups untouched ---
    ...options.filter(option => option.id !== "cut"), // Keep default copy/paste etc.
    rowIds.length === 1 && [
      {
        id: "insert-row-before",
        label: "Insert row before",
        handler: () => store.insertRowBefore(rowIds[0])
      },
      {
        id: "insert-row-after",
        label: "Insert row after",
        handler: () => store.insertRowAfter(rowIds[0])
      }
    ],
    colIds.length === 1 && [
      {
        id: "insert-column-before",
        label: "Insert column before",
        handler: () => store.insertColumnBefore(colIds[0])
      },
      {
        id: "insert-column-after",
        label: "Insert column after",
        handler: () => store.insertColumnAfter(colIds[0])
      }
    ],
    !isEmpty(rowIds) && [
      {
        id: "rerun-rows",
        label: `Rerun ${plur("row", rowIds)}`,
        handler: () => store.rerunRows(rowIds)
      },
      {
        id: "clear-rows",
        label: `Clear ${plur("row", rowIds)}`,
        handler: () => store.clearRows(rowIds)
      },
      {
        id: "delete-rows",
        label: `Delete ${plur("row", rowIds)}`,
        handler: () => store.deleteRows(rowIds)
      }
    ],
    !isEmpty(colIds) && [
      {
        id: "rerun-columns",
        label: `Rerun ${plur("column", colIds)}`,
        handler: () => store.rerunColumns(colIds)
      },
      {
        id: "clear-columns",
        label: `Clear ${plur("column", colIds)}`,
        handler: () => store.clearColumns(colIds)
      },
      {
        id: "delete-columns",
        label: `Delete ${plur("column", colIds)}`,
        handler: () => store.deleteColumns(colIds)
      }
    ]
  ]);
};

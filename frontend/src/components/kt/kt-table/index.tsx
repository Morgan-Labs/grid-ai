import { useMemo, useState } from "react";
import { Column, ReactGrid, Row } from "@silevis/reactgrid";
import { BoxProps, ScrollArea, Pagination, Group, Text, Select, Stack, Box, ComboboxItem } from "@mantine/core";
import {
  Cell,
  handleCellChange,
  handleContextMenu,
  HEADER_ROW_ID,
  SOURCE_COLUMN_ID
} from "./index.utils";
import {
  KtCell,
  KtCellTemplate,
  KtColumnCell,
  KtColumnCellTemplate,
  KtRowCellTemplate
} from "./kt-cells";
import { KtProgressBar } from "../kt-progress-bar";
import { useStore } from "@config/store";
import { cn } from "@utils/functions";
import classes from "./index.module.css";

const PAGE_SIZES = [10, 25, 50, 100];

export function KtTable(props: BoxProps) {
  const table = useStore(store => store.getTable());
  const columns = table.columns;
  const rows = table.rows;
  
  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);
  
  // Use a custom property on the table to store the Document column width
  // @ts-ignore - Adding a custom property to the table object
  const sourceColumnWidth = table.sourceColumnWidth || 350; // Increased default width
  const visibleColumns = useMemo(
    () => columns.filter(column => !column.hidden),
    [columns]
  );
  
  // First filter by hidden state, then apply pagination
  const filteredRows = useMemo(() => rows.filter(row => !row.hidden), [rows]);
  
  // Calculate total pages
  const totalRows = filteredRows.length;
  const totalPages = Math.max(1, Math.ceil(totalRows / pageSize));
  
  // Ensure current page is in valid range when total changes
  const safeCurrentPage = Math.min(currentPage, totalPages);
  if (safeCurrentPage !== currentPage) {
    setCurrentPage(safeCurrentPage);
  }
  
  // Apply pagination to filtered rows
  const visibleRows = useMemo(() => {
    const startIndex = (safeCurrentPage - 1) * pageSize;
    return filteredRows.slice(startIndex, startIndex + pageSize);
  }, [filteredRows, safeCurrentPage, pageSize]);

  const gridColumns = useMemo<Column[]>(
    () => [
      { columnId: SOURCE_COLUMN_ID, width: sourceColumnWidth, resizable: true },
      ...visibleColumns.map(column => ({
        columnId: column.id,
        width: column.width,
        resizable: true
      }))
    ],
    [visibleColumns, sourceColumnWidth]
  );

  const gridRows = useMemo<Row<Cell>[]>(
    () => [
      {
        rowId: HEADER_ROW_ID,
        cells: [
          { type: "header", text: "Document" },
          ...visibleColumns.map<KtColumnCell>((column, index) => ({
            type: "kt-column",
            column,
            columnIndex: index
          }))
        ]
      },
      ...visibleRows.map<Row<Cell>>(row => ({
        rowId: row.id,
        height: 48,
        cells: [
          { type: "kt-row", row },
          ...visibleColumns.map<KtCell>(column => ({
            type: "kt-cell",
            column,
            row,
            cell: row.cells[column.id]
          }))
        ]
      }))
    ],
    [visibleRows, visibleColumns]
  );

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  const handlePageSizeChange = (value: string | null, _option: ComboboxItem) => {
    if (value === null) return;
    const newPageSize = parseInt(value, 10);
    setPageSize(newPageSize);
    // Reset to first page when changing page size to avoid out-of-range issues
    setCurrentPage(1);
  };

  return (
    <Stack gap="sm" pb={0} {...props}>
      <KtProgressBar />
      <ScrollArea
        style={{ flex: 1 }}
        className={cn(classes.reactGridWrapper, props.className)}
      >
        <ReactGrid
          enableRangeSelection
          enableColumnSelection
          enableRowSelection
          minColumnWidth={100}
          columns={gridColumns}
          rows={gridRows}
          onContextMenu={handleContextMenu}
          onCellsChanged={handleCellChange}
          onColumnResized={(columnId, width) => {
            if (columnId === SOURCE_COLUMN_ID) {
              // Update the custom property in the store
              useStore.getState().editActiveTable({
                // @ts-ignore - Adding a custom property to the table object
                sourceColumnWidth: width
              });
            } else {
              useStore.getState().editColumn(String(columnId), { width });
            }
          }}
          customCellTemplates={{
            "kt-cell": new KtCellTemplate(),
            "kt-column": new KtColumnCellTemplate(),
            "kt-row": new KtRowCellTemplate()
          }}
        />
      </ScrollArea>
      
      {totalRows > 10 && (
        <Box px="md" py="xs" className={classes.paginationContainer}>
          <Group justify="space-between" align="center">
            <Text size="sm" color="dimmed">
              Showing {visibleRows.length} of {totalRows} rows
            </Text>
            <Group gap="xs">
              <Select
                value={pageSize.toString()}
                onChange={handlePageSizeChange}
                data={PAGE_SIZES.map(size => ({ value: size.toString(), label: `${size} / page` }))}
                size="xs"
                style={{ width: 110 }}
              />
              <Pagination
                value={safeCurrentPage}
                onChange={handlePageChange}
                total={totalPages}
                size="sm"
                withEdges
              />
            </Group>
          </Group>
        </Box>
      )}
    </Stack>
  );
}

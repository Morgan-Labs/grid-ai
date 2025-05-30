import { useMemo, useState, useEffect, useCallback } from "react";
import { Column, ReactGrid, Row } from "@silevis/reactgrid";
import { BoxProps, ScrollArea, Pagination, Group, Text, Select, Stack, Box, ComboboxItem, LoadingOverlay } from "@mantine/core";
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
  const isAuthenticated = useStore(state => state.auth.isAuthenticated);
  const [isLoading, setIsLoading] = useState(true);
  
  // Add effect to handle loading state
  useEffect(() => {
    if (isAuthenticated) {
      // Set loading to false after a short delay to ensure smooth transition
      const timer = setTimeout(() => {
        setIsLoading(false);
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [isAuthenticated]);

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
  
  // Calculate the safe current page
  const safeCurrentPage = useMemo(() => {
    return Math.min(currentPage, totalPages);
  }, [currentPage, totalPages]);

  // Use useEffect to handle page validation to avoid state updates during render
  useEffect(() => {
    if (safeCurrentPage !== currentPage) {
      setCurrentPage(safeCurrentPage);
    }
  }, [safeCurrentPage, currentPage]);

  // Apply pagination to filtered rows using the safe current page
  const visibleRows = useMemo(() => {
    const startIndex = (safeCurrentPage - 1) * pageSize;
    const endIndex = startIndex + pageSize;
    return filteredRows.slice(startIndex, endIndex);
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
          { 
            type: "header", 
            text: "📄 Document"
          },
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

  // Function to find and navigate to a specific row
  const navigateToRow = useCallback((rowId: string) => {
    const rowIndex = filteredRows.findIndex(row => row.id === rowId);
    if (rowIndex !== -1) {
      const targetPage = Math.ceil((rowIndex + 1) / pageSize);
      if (targetPage !== currentPage) {
        setCurrentPage(targetPage);
        return true; // Indicate page change happened
      }
    }
    return false; // No page change needed
  }, [filteredRows, pageSize, currentPage]);

  // Store the navigation function in the Zustand store for type-safe access
  useEffect(() => {
    useStore.setState({ navigateToRow });
    
    // Cleanup function to remove the navigation function when component unmounts
    return () => {
      useStore.setState({ navigateToRow: null });
    };
  }, [navigateToRow]);

  return (
    <Stack gap="sm" pb={0} {...props}>
      <KtProgressBar />
      <Box pos="relative">
        <LoadingOverlay 
          visible={isLoading} 
          zIndex={1000} 
          overlayProps={{ blur: 2 }}
          loaderProps={{ type: 'dots' }}
        />
        <ScrollArea
          key={`table-${totalRows}-${currentPage}-${pageSize}`}
          style={{ flex: 1, height: '100%' }}
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
      </Box>
      
      {totalRows > 10 && (
        <Box px="md" py="xs" className={classes.paginationContainer}>
          <Group justify="space-between" align="center">
            <Text size="sm" color="dimmed">
              Showing {((currentPage - 1) * pageSize) + 1}-{Math.min(currentPage * pageSize, totalRows)} of {totalRows} rows
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
                value={currentPage}
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

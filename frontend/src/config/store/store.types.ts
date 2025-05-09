import { z } from "zod";
import { answerSchema, documentSchema, chunkSchema } from "../api";

export interface AuthState {
  token: string | null;
  isAuthenticated: boolean;
  isAuthenticating: boolean;
}

export interface Store {
  colorScheme: "light" | "dark";
  tables: AnswerTable[];
  activeTableId: string;
  activePopoverId: string | null;
  documentPreviews: Record<string, string[]>; // Store document preview content by document ID
  auth: AuthState;
  _saveTableStateTimer: ReturnType<typeof setTimeout> | null; // Timer for debouncing table state saves
  isLoading: boolean;
  savedStates: TableStateListItem[]; // Added: To store the list of fetched states

  toggleColorScheme: () => void;
  setActivePopover: (id: string | null) => void;
  
  // CSV/Excel import
  importCsvData: (data: string[][], preserveExistingColumns?: boolean) => Promise<void>;

  // Authentication actions
  login: (password: string) => Promise<void>;
  logout: () => void;
  checkAuth: () => Promise<{ isValid: boolean; servicesInitialized: boolean } | false>;

  getTable: (id?: string) => AnswerTable;
  addTable: (name: string) => void;
  editTable: (id: string, table: Partial<AnswerTable>) => void;
  editActiveTable: (table: Partial<AnswerTable>) => void;
  switchTable: (id: string) => void;
  deleteTable: (id: string) => void;

  insertColumnBefore: (id?: string) => void;
  insertColumnAfter: (id?: string) => void;
  editColumn: (id: string, column: Partial<AnswerTableColumn>) => void;
  rerunColumns: (ids: string[]) => void;
  clearColumns: (ids: string[]) => void;
  unwindColumn: (id: string) => void;
  toggleAllColumns: (hidden: boolean) => void;
  deleteColumns: (ids: string[]) => void;
  reorderColumns: (sourceIndex: number, targetIndex: number) => void;

  insertRowBefore: (id?: string) => void;
  insertRowAfter: (id?: string) => void;
  // Updated fillRow signature to accept optional options
  fillRow: (id: string, file: File, options?: { showNotification?: boolean }) => Promise<void>;
  fillRows: (files: File[]) => Promise<void>;
  // Add option to suppress progress notification
  rerunRows: (ids: string[], options?: { suppressInProgressNotification?: boolean }) => void;
  clearRows: (ids: string[]) => void;
  deleteRows: (ids: string[]) => void;

  editCells: (
    cells: { rowId: string; columnId: string; cell: CellValue }[],
    tableId?: string
  ) => void;
  // Add option to suppress progress notification
  rerunCells: (cells: { rowId: string; columnId: string }[], options?: { suppressInProgressNotification?: boolean }) => void;
  clearCells: (cells: { rowId: string; columnId: string }[]) => void;

  addGlobalRules: (rules: Omit<AnswerTableGlobalRule, "id">[]) => void;
  editGlobalRule: (id: string, rule: Partial<AnswerTableGlobalRule>) => void;
  deleteGlobalRules: (ids?: string[]) => void;

  openChunks: (cells: { rowId: string; columnId: string }[]) => void;
  closeChunks: () => void;

  addFilter: (filter: Omit<AnswerTableFilter, "id">) => void;
  editFilter: (id: string, filter: Partial<AnswerTableFilter>) => void;
  deleteFilters: (ids?: string[]) => void;
  applyFilters: () => void;

  // Add document preview content to the store
  addDocumentPreview: (documentId: string, content: string[]) => void;

  // Ingest single document by ID (for context menu)
  ingestSingleDocumentById: (documentId: string, rowId: string, originalColumnId: string) => Promise<{ success: boolean }>;
  
  // Table state persistence
  saveTableState: () => Promise<void>;
  loadSavedStatesAndActivateLatest: () => Promise<void>; // Added
  loadTableState: (tableId: string) => Promise<void>; // Added
  setTableData: (tableId: string, fullTableDataContainer: { name?: string, data: Omit<AnswerTable, 'id' | 'name'> }) => void; // Added
  
  clear: (allTables?: boolean) => void;
}

export interface ResolvedEntity {
  original: string | string[];  // Allow both string and array of strings
  resolved: string | string[];  // Allow both string and array of strings
  fullAnswer: string;
  entityType: string;
  source: {
    type: 'global' | 'column';
    id: string;
  };
}

export interface RequestProgress {
  total: number;
  completed: number;
  inProgress: boolean;
  error?: boolean;
}

export interface AnswerTable {
  id: string;
  name: string;
  columns: AnswerTableColumn[];
  rows: AnswerTableRow[];
  globalRules: AnswerTableGlobalRule[];
  filters: AnswerTableFilter[];
  chunks: Record<CellKey, Chunk[]>;
  openedChunks: CellKey[];
  loadingCells: Record<CellKey, true>;
  uploadingFiles: boolean;
  requestProgress?: RequestProgress;
}

export interface AnswerTableColumn {
  id: string;
  width: number;
  hidden: boolean;
  entityType: string;
  type: "int" | "str" | "bool" | "int_array" | "str_array" | "url";
  generate: boolean;
  query: string;
  rules: AnswerTableRule[];
  resolvedEntities?: ResolvedEntity[];
  llmModel?: string;     // gpt-4o, claude-3-5-sonnet, gemini-1.5-pro, etc.
}

export interface AnswerTableRow {
  id: string;
  sourceData: SourceData | null;
  hidden: boolean;
  cells: Record<string, CellValue>;
}

export interface AnswerTableFilter {
  id: string;
  columnId: string;
  criteria: "contains" | "contains_not";
  value: string;
}

export interface AnswerTableGlobalRule extends AnswerTableRule {
  id: string;
  entityType: string;
  resolvedEntities?: ResolvedEntity[];
}

export interface AnswerTableRule {
  type: "must_return" | "may_return" | "max_length" | "resolve_entity";
  options?: string[];
  length?: number;
}

export type SourceData = 
  | {
      type: "document";
      document: Document;
    }
  | {
      type: "loading";
      name: string;
    }
  | {
      type: "error";
      name: string;
      error: string;
    };

export type CellKey = `${string}-${string}`;
export type Document = z.infer<typeof documentSchema>;
export type CellValue = z.infer<typeof answerSchema> | undefined;
export type Chunk = z.infer<typeof chunkSchema>;

// --- ADD API-SPECIFIC TABLE STATE TYPES ---

// For the items in the list returned by GET /table-state/
export interface TableStateListItem {
  id: string;
  name: string;
  created_at: string; // Or Date, if you parse it
  updated_at: string; // Or Date
  user_id?: string; // Optional, if your backend provides it
}

// For the overall response of GET /table-state/
export interface TableStateListResponse {
  items: TableStateListItem[];
  // include other pagination fields if your API supports them e.g.
  // total?: number;
  // page?: number;
  // size?: number;
}

// For the response of GET /table-state/{id} (a single full table state)
// Also generally represents the TableState model on the backend.
export interface TableState {
  id: string;
  name: string;
  // The 'data' field contains the actual table structure, similar to AnswerTable.
  // For simplicity, we can say 'data' IS an AnswerTable, but without its own id/name if those are redundant
  // with the outer TableState id/name.
  // However, your backend TableState.data might have a structure identical to AnswerTable including id/name.
  // Let's assume backend's TableState.data *is* the AnswerTable structure for now.
  data: AnswerTable; 
  created_at: string; // Or Date
  updated_at: string; // Or Date
  user_id?: string; // Optional
}

// Type alias for clarity, as GET /table-state/{id} returns a TableState object
export type TableStateResponse = TableState;

// For the payload of POST /table-state/
export interface TableStateCreate {
  id: string;       // Usually CUID generated on frontend
  name: string;
  data: AnswerTable; // The full table data structure
}

// For the payload of PUT /table-state/{id}
export interface TableStateUpdate {
  name?: string;      // Optional: only if name is being changed
  data?: Partial<AnswerTable>; // Optional: allow partial or full update of the data field
}

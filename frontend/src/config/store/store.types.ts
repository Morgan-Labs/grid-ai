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
  documents: Record<string, Document>; // Store documents and their statuses
  auth: AuthState;
  _saveTableStateTimer: ReturnType<typeof setTimeout> | null; // Timer for debouncing table state saves
  navigateToRow: ((rowId: string) => boolean) | null; // Function to navigate to a specific row

  toggleColorScheme: () => void;
  setActivePopover: (id: string | null) => void;
  
  // Document methods
  addDocument: (document: Document) => void;
  updateDocumentStatus: (documentId: string, status: string) => void;
  checkDocumentStatus: (documentId: string) => Promise<void>;
  pollDocumentStatus: (documentId: string, interval?: number, maxAttempts?: number) => void;
  
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
  loadLatestTableState: () => Promise<void>;
  
  clear: (allTables?: boolean) => void;

  // Loading state
  isLoading: boolean;
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

import { HTMLAttributes, ReactNode, RefObject, useEffect, useRef } from "react";
import { Uncertain } from "@silevis/reactgrid";
import { Text, Badge, Box } from "@mantine/core";
import { isArray, isBoolean } from "lodash-es";
import { KtCell } from "./kt-cell";
import { CellValue } from "@config/store";
import { niceTry } from "@utils/functions";

// Utility

export function isKtCell(cell: Uncertain<KtCell>): cell is KtCell {
  return Boolean(cell.type === "kt-cell" && cell.column && cell.row);
}

// URL regex pattern for detecting URLs in text
const URL_REGEX = /(https?:\/\/[^\s<>"']+|www\.[^\s<>"']+)/g;

// Special Salesforce URL pattern (like the one in the example)
const SF_URL_REGEX = /https:\/\/[^\s<>"']+force\.com\/[^\s<>"']+\/[a-zA-Z0-9]{15,18}\/[^\s<>"']+/g;


// Function to detect if a string contains a URL
function containsUrl(text: string): boolean {
  return URL_REGEX.test(text) || SF_URL_REGEX.test(text);
}

// Function to render a hyperlink from text
function renderHyperlink(text: string): ReactNode {
  // Reset the regex lastIndex property to ensure consistent behavior
  URL_REGEX.lastIndex = 0;
  SF_URL_REGEX.lastIndex = 0;
  
  try {
    // First check for Salesforce-specific URLs
    if (SF_URL_REGEX.test(text)) {
      SF_URL_REGEX.lastIndex = 0; // Reset for the next search
      const match = SF_URL_REGEX.exec(text)?.[0];
      if (match) {
        return (
          <Text 
            style={{ color: '#228be6', cursor: 'pointer', textDecoration: 'underline' }}
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation(); // Prevent cell selection
              window.open(match, '_blank', 'noopener,noreferrer'); // Manually open in new tab
            }}
          >
            View Matter
          </Text>
        );
      }
    }
    
    // For standard URLs, check if it's a complete URL that can be rendered as a link
    URL_REGEX.lastIndex = 0; // Reset for the next search
    if (URL_REGEX.test(text)) {
      URL_REGEX.lastIndex = 0; // Reset for the next search
      const match = URL_REGEX.exec(text)?.[0];
      if (match) {
        const url = match.startsWith('http') ? match : `https://${match}`;
        return (
          <Text 
            style={{ color: '#228be6', cursor: 'pointer', textDecoration: 'underline' }}
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation(); // Prevent cell selection
              window.open(url, '_blank', 'noopener,noreferrer'); // Manually open in new tab
            }}
          >
            {match}
          </Text>
        );
      }
    }
  } catch (error) {
    console.error("Error rendering hyperlink:", error);
    // In case of any error, fall back to plain text
    return <Text>{text}</Text>;
  }
  
  // If not a URL, return the text as is
  return text;
}

export function formatCell(cell?: CellValue): ReactNode {
  if (cell === undefined) {
    return null;
  } else if (cell === null) {
    return <Text c="dimmed">Not found</Text>;
  } else if (isBoolean(cell)) {
    return <Badge>{String(cell)}</Badge>;
  } else {
    const cellText = isArray(cell) ? cell.join(", ") : String(cell);
    
    // Check if the cell content might contain a URL
    if (typeof cellText === 'string' && containsUrl(cellText)) {
      return (
        <Text lineClamp={2}>
          {renderHyperlink(cellText)}
        </Text>
      );
    } else {
      return (
        <Text lineClamp={2}>
          {cellText}
        </Text>
      );
    }
  }
}

// Editor wrapper

type EditorWrapperProps = {
  defaultValue: CellValue;
  onChange: (value: CellValue, commit?: boolean) => void;
  children: (
    inputProps: InputProps,
    handleChange: (value: CellValue) => void
  ) => ReactNode;
};

type InputProps = Pick<
  HTMLAttributes<HTMLElement>,
  "onCopy" | "onCut" | "onPaste"
> & { ref: RefObject<any> };

export function EditorWrapper({
  defaultValue,
  onChange,
  children
}: EditorWrapperProps) {
  const inputRef = useRef<any>(null);
  const escapedPressed = useRef(false);
  const lastChange = useRef(defaultValue);

  const handleChange = (value: CellValue) => {
    lastChange.current = value;
    onChange(value);
  };

  const inputProps: InputProps = {
    ref: inputRef,
    onCopy: e => e.stopPropagation(),
    onCut: e => e.stopPropagation(),
    onPaste: e => e.stopPropagation()
  };

  useEffect(
    () =>
      niceTry(() => {
        const input = inputRef.current;
        if (!input) return;
        input.focus();
        input.setSelectionRange(input.value.length, input.value.length);
      }),
    []
  );

  return (
    <Box
      w="100%"
      mih="100%"
      tabIndex={0}
      onPointerDown={e => e.stopPropagation()}
      onBlur={() => {
        onChange(lastChange.current, !escapedPressed.current);
        escapedPressed.current = false;
      }}
      onKeyDown={e => {
        if (e.key === "Escape") {
          escapedPressed.current = true;
        }
        if (!["Escape", "Enter"].includes(e.key)) {
          e.stopPropagation();
        }
      }}
    >
      {children(inputProps, handleChange)}
    </Box>
  );
}

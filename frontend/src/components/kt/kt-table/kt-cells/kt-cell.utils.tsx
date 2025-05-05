import React, { HTMLAttributes, ReactNode, RefObject, useEffect, useRef } from "react";
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

// Function to create a clickable link component
function createLinkComponent(url: string, displayText: string): ReactNode {
  const fullUrl = url.startsWith('http') ? url : `https://${url}`;
  return (
    <Text 
      component="span"
      style={{ color: '#228be6', cursor: 'pointer', textDecoration: 'underline' }}
      onClick={(e) => {
        e.preventDefault();
        e.stopPropagation(); // Prevent cell selection
        window.open(fullUrl, '_blank', 'noopener,noreferrer'); // Manually open in new tab
      }}
    >
      {displayText}
    </Text>
  );
}

// Function to render text with embedded hyperlinks
function renderTextWithLinks(text: string): ReactNode {
  if (!text || typeof text !== 'string') {
    return text;
  }

  try {
    // Check for Salesforce URLs
    SF_URL_REGEX.lastIndex = 0;
    let sfMatch = SF_URL_REGEX.exec(text);
    if (sfMatch) {
      const parts = [];
      let lastIndex = 0;
      
      while (sfMatch) {
        // Add text before the match
        if (sfMatch.index > lastIndex) {
          parts.push(text.substring(lastIndex, sfMatch.index));
        }
        
        // Add the link component
        parts.push(createLinkComponent(sfMatch[0], 'View URL'));
        
        lastIndex = sfMatch.index + sfMatch[0].length;
        SF_URL_REGEX.lastIndex = lastIndex;
        sfMatch = SF_URL_REGEX.exec(text);
      }
      
      // Add any remaining text
      if (lastIndex < text.length) {
        parts.push(text.substring(lastIndex));
      }
      
      // Return the array of text and link components
      return (
        <>
          {parts.map((part, i) => (
            <React.Fragment key={i}>{part}</React.Fragment>
          ))}
        </>
      );
    }
    
    // Check for regular URLs
    URL_REGEX.lastIndex = 0;
    let urlMatch = URL_REGEX.exec(text);
    if (urlMatch) {
      const parts = [];
      let lastIndex = 0;
      
      while (urlMatch) {
        // Add text before the match
        if (urlMatch.index > lastIndex) {
          parts.push(text.substring(lastIndex, urlMatch.index));
        }
        
        // Add the link component
        parts.push(createLinkComponent(urlMatch[0], urlMatch[0]));
        
        lastIndex = urlMatch.index + urlMatch[0].length;
        URL_REGEX.lastIndex = lastIndex;
        urlMatch = URL_REGEX.exec(text);
      }
      
      // Add any remaining text
      if (lastIndex < text.length) {
        parts.push(text.substring(lastIndex));
      }
      
      // Return the array of text and link components
      return (
        <>
          {parts.map((part, i) => (
            <React.Fragment key={i}>{part}</React.Fragment>
          ))}
        </>
      );
    }
  } catch (error) {
    console.error("Error rendering hyperlinks:", error);
    // In case of error, return the original text
    return text;
  }
  
  // No URLs found, return the original text
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
          {renderTextWithLinks(cellText)}
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

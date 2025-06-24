import { ReactNode, useMemo, useCallback, ClipboardEvent } from "react";
import { Box, Text, Input, InputWrapperProps, Paper } from "@mantine/core";
import { useUncontrolled } from "@mantine/hooks";
import styled from "@emotion/styled";
import {
  MentionsInput,
  Mention as ReactMention,
  SuggestionDataItem
} from "react-mentions";
import { cn } from "@utils/functions";
import classes from "./index.module.css";

interface Props extends Omit<InputWrapperProps, "onChange"> {
  placeholder?: string;
  disabled?: boolean;
  value?: string;
  defaultValue?: string;
  onChange: (value: string) => void;
  options: Array<{
    trigger: string;
    data: SuggestionDataItem[];
    color?: (item: SuggestionDataItem) => string;
    render?: (item: SuggestionDataItem) => ReactNode;
  }>;
}

export function Mention({
  placeholder,
  disabled,
  value: value_,
  defaultValue,
  onChange,
  options,
  ...props
}: Props): JSX.Element {
  const [value, setValue] = useUncontrolled({
    value: value_,
    defaultValue,
    onChange
  });

  const colors = useMemo(() => {
    try {
      if (!value) return [];
      // Updated regex to match the correct format: @[Display](id)
      const matches = [...value.matchAll(/@\[[^\]]+\]\(([^)]+)\)/g)];
      
      return matches.map(match => {
        try {
          const group = options.find(option => match[0].startsWith(option.trigger));
          const option = group?.data.find(item => item.id === match[1]);
          return (option && group?.color?.(option)) || "transparent";
        } catch (error) {
          return "transparent";
        }
      });
    } catch (error) {
      return [];
    }
  }, [value, options]);

  // Create a memoized handler for onChange to prevent unnecessary re-renders
  const handleChange = useCallback((event: { target: { value: string } }) => {
    try {
      setValue(event.target.value);
    } catch (error) {
      // Fallback to empty string if there's an error
      setValue('');
    }
  }, [setValue]);

  // Handle paste events to preserve mention formatting
  const handlePaste = useCallback((event: ClipboardEvent<HTMLTextAreaElement>) => {
    try {
      const clipboardData = event.clipboardData;
      if (!clipboardData) return;

      // Get both plain text and HTML from clipboard
      const plainText = clipboardData.getData('text/plain');
      const htmlText = clipboardData.getData('text/html');

      // Check if the pasted content contains mention markup
      const mentionPattern = /@\[[^\]]+\]\([^)]+\)/g;
      
      // If HTML contains mention-like structure, let react-mentions handle it
      if (htmlText && (htmlText.includes('data-mention') || mentionPattern.test(htmlText))) {
        return;
      }

      // If plain text contains mention markup, let react-mentions handle it
      if (plainText && mentionPattern.test(plainText)) {
        return;
      }

      // Smart reconstruction: Check if pasted text matches any column entity types
      if (plainText && options.length > 0) {
        const triggerGroup = options.find(opt => opt.trigger === '@');
        if (triggerGroup) {
          // Check if any part of the pasted text matches column entity types
          let hasMatchingEntity = false;

          triggerGroup.data.forEach(column => {
            const entityName = column.display;
            if (entityName && plainText.includes(entityName)) {
              hasMatchingEntity = true;
            }
          });

          if (hasMatchingEntity) {
            // Let react-mentions handle the paste normally, then process the result
            // The processing will happen in the parent component via the smart processing logic
            return;
          }
        }
      }
    } catch (error) {
      console.error('Error handling paste in Mention component:', error);
    }
  }, [options]);

  // Handle copy events to ensure proper mention formatting
  const handleCopy = useCallback((_event: ClipboardEvent<HTMLTextAreaElement>) => {
    try {
      const selection = window.getSelection();
      if (!selection || selection.rangeCount === 0) return;

      const range = selection.getRangeAt(0);
      const selectedText = range.toString();
      
      // Check if the selection contains mentions
      const mentionPattern = /@\[[^\]]+\]\([^)]+\)/g;
      
      if (selectedText && value && mentionPattern.test(value)) {
        // Find mentions in the current value that overlap with selection
        const mentions = [...value.matchAll(mentionPattern)];
        const selectionStart = value.indexOf(selectedText);
        
        if (selectionStart >= 0) {
          // Check if any mentions are within the selection
          const overlappingMentions = mentions.filter(match => {
            const mentionStart = match.index || 0;
            const mentionEnd = mentionStart + match[0].length;
            const selectionEnd = selectionStart + selectedText.length;
            
            return (mentionStart < selectionEnd && mentionEnd > selectionStart);
          });
          
          if (overlappingMentions.length > 0) {
            // Let react-mentions handle the copy with proper markup
            return;
          }
        }
      }
    } catch (error) {
      console.error('Error handling copy in Mention component:', error);
    }
  }, [value]);

  // Create a memoized handler for suggestions container to prevent unnecessary re-renders
  const renderSuggestionsContainer = useCallback((node: React.ReactNode) => (
    <Paper 
      withBorder 
      shadow="sm" 
      p={4}
      style={{ zIndex: 1001 }}
    >
      {node}
    </Paper>
  ), []);

  return (
    <StyledInputWrapper
      {...props}
      colors={colors}
      className={cn(classes.wrapper, props.className)}
    >
      <MentionsInput
        disabled={disabled}
        allowSpaceInQuery
        allowSuggestionsAboveCursor
        placeholder={placeholder}
        value={value || ''}
        onChange={handleChange}
        className="mentions"
        a11ySuggestionsListLabel="Suggested mentions"
        customSuggestionsContainer={renderSuggestionsContainer}
        onPaste={handlePaste}
        onCopy={handleCopy}
      >
        {options.map(({ trigger, data, render }) => (
          <ReactMention
            key={trigger}
            trigger={trigger}
            data={data}
            appendSpaceOnAdd
            markup={`${trigger}[__display__](__id__)`}
            renderSuggestion={(suggestion, _, __, ___, active) => (
              <Box className={cn(classes.suggestion, active && classes.active)}>
                {render ? (
                  render(suggestion)
                ) : (
                  <Text>{suggestion.display}</Text>
                )}
              </Box>
            )}
          />
        ))}
      </MentionsInput>
    </StyledInputWrapper>
  );
}

const StyledInputWrapper = styled(
  ({ colors, ...props }: InputWrapperProps & { colors: string[] }) => (
    <Input.Wrapper {...props} />
  )
)`
  .mentions__highlighter {
    ${({ colors }) =>
      colors.map(
        (color, index) => `
        > strong:nth-of-type(${index + 1}) {
          background-color: ${color};
        }
      `
      )}
  }
`;

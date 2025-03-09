import { ReactNode, useMemo, useCallback } from "react";
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
}: Props) {
  const [value, setValue] = useUncontrolled({
    value: value_,
    defaultValue,
    onChange
  });

  const colors = useMemo(() => {
    try {
      if (!value) return [];
      const matches = [...value.matchAll(/.\[[^\]]+\]\((\w+)\)/g)];
      return matches.map(match => {
        try {
          const group = options.find(option => match[0].startsWith(option.trigger));
          const option = group?.data.find(item => item.id === match[1]);
          return (option && group?.color?.(option)) || "transparent";
        } catch (error) {
          console.error('Error processing match in Mention component:', error);
          return "transparent";
        }
      });
    } catch (error) {
      console.error('Error calculating colors in Mention component:', error);
      return [];
    }
  }, [value, options]);

  // Create a memoized handler for onChange to prevent unnecessary re-renders
  const handleChange = useCallback((event: { target: { value: string } }) => {
    try {
      setValue(event.target.value);
    } catch (error) {
      console.error('Error in Mention component onChange:', error);
      // Fallback to empty string if there's an error
      setValue('');
    }
  }, [setValue]);

  // Create a memoized handler for suggestions container to prevent unnecessary re-renders
  const renderSuggestionsContainer = useCallback((node: React.ReactNode) => (
    <Paper withBorder shadow="sm" p={4}>
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

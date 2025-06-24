import { ColorSwatch, Group, InputWrapperProps, Text } from "@mantine/core";
import { useStore } from "@config/store";
import { Mention } from "@components";
import { entityColor } from "@utils/functions";
import { useMemo } from "react";

interface Props extends Omit<InputWrapperProps, "onChange"> {
  placeholder?: string;
  disabled?: boolean;
  value?: string;
  defaultValue?: string;
  onChange: (value: string) => void;
}

// Example questions based on column type
const getExampleQuestions = (entityType: string, type: string): string[] => {
  const baseType = type.includes('array') ? type.replace('_array', '') : type;
  
  switch (baseType) {
    case 'str':
      return [
        `What is the ${entityType.toLowerCase()}?`,
        `Extract the ${entityType.toLowerCase()} from the document`,
        `Find the ${entityType.toLowerCase()} mentioned in the text`
      ];
    case 'int':
      return [
        `What is the ${entityType.toLowerCase()} value?`,
        `Extract the ${entityType.toLowerCase()} number from the document`,
        `Calculate the total ${entityType.toLowerCase()}`
      ];
    case 'bool':
      return [
        `Is the ${entityType.toLowerCase()} present?`,
        `Does the document mention ${entityType.toLowerCase()}?`,
        `Check if ${entityType.toLowerCase()} is applicable`
      ];
    default:
      return [
        `What is the ${entityType.toLowerCase()}?`,
        `Extract the ${entityType.toLowerCase()} from the document`
      ];
  }
};

export function KtColumnQuestion({
  placeholder,
  disabled,
  value,
  defaultValue,
  onChange,
  ...props
}: Props) {
  const columns = useStore(store => store.getTable().columns);

  // Process the value to convert plain entity names to mention format
  const processedValue = useMemo(() => {
    const currentValue = value || defaultValue || '';
    if (!currentValue) return currentValue;

    let processed = currentValue;
    
    // Find columns with entityTypes that appear in the text but aren't in mention format
    columns
      .filter(column => column.entityType.trim())
      .forEach(column => {
        const entityType = column.entityType;
        // Look for the entity name that's not already in mention format
        const regex = new RegExp(`\\b${entityType}\\b(?!\\])`, 'gi');
        processed = processed.replace(regex, `@[${entityType}](${column.id})`);
      });

    return processed;
  }, [value, defaultValue, columns]);
  const currentColumn = useMemo(() => {
    // Try to find the current column based on the defaultValue or value
    // This is a heuristic and might not always work perfectly
    return columns.find(col => 
      col.query === defaultValue || col.query === value
    );
  }, [columns, defaultValue, value]);
  
  // Generate example placeholder based on column type if available
  const examplePlaceholder = useMemo(() => {
    if (!currentColumn?.entityType) return placeholder;
    
    const examples = getExampleQuestions(
      currentColumn.entityType, 
      currentColumn.type
    );
    
    // Return a random example from the list
    return examples[Math.floor(Math.random() * examples.length)];
  }, [currentColumn, placeholder]);
  
  return (
    <>
      <Mention
        required
        placeholder={examplePlaceholder}
        disabled={disabled}
        value={processedValue}
        defaultValue={processedValue}
        onChange={onChange}
        options={[
          {
            trigger: "@",
            data: columns
              .filter(column => column.entityType.trim())
              .map(column => ({
                id: column.id,
                display: column.entityType
              })),
            color: item => entityColor(item.display ?? "").fill,
            render: item => {
              const color = entityColor(item.display ?? "").fill;
              return (
                <Group>
                  <ColorSwatch size={12} color={color} />
                  <span>{item.display}</span>
                </Group>
              );
            }
          }
        ]}
        {...props}
      />
      {!value && !defaultValue && (
        <Text size="xs" c="dimmed" mt={5}>
          Examples: "What is the invoice number?", "Extract the total amount", "Find the customer name"
        </Text>
      )}
    </>
  );
}

import { Text, Tooltip } from "@mantine/core";
import { listTableStates } from "../../../services/api/table-state";
import { TableStateListItem } from "../../../config/store/store.types";
import { useStore } from "@config/store";
import { useState, useEffect } from "react";

export function KtTimestamp() {
  const table = useStore(store => store.getTable());
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);

  useEffect(() => {
    const fetchLastUpdated = async () => {
      try {
        const response = await listTableStates();
        const tableState = response.items.find((item: TableStateListItem) => item.id === table.id);
        if (tableState) {
          setLastUpdated(tableState.updated_at);
        }
      } catch (error) {
        console.error('Failed to fetch table state:', error);
      }
    };
    
    fetchLastUpdated();
  }, [table.id]);

  if (!lastUpdated) return null;

  const date = new Date(lastUpdated + 'Z');
  const shortDate = date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  });
  const fullDate = date.toLocaleString('en-US', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    timeZoneName: 'short'
  });

  return (
    <Tooltip label={fullDate}>
      <Text size="sm" c="dimmed">
        Updated: {shortDate}
      </Text>
    </Tooltip>
  );
} 
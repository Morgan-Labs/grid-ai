import { Text, Tooltip } from "@mantine/core";
import { useStore } from "@config/store";
import { listTableStates, TableStateApiResponse } from "../../../services/api/table-state";
import { useState, useEffect } from "react";
import dayjs from "dayjs";

// interface KtTimestampProps {
//   // Add any additional props here if needed
// }

export function KtTimestamp() {
  const table = useStore(store => store.getTable());
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);

  useEffect(() => {
    const fetchLastUpdated = async () => {
      try {
        const response = await listTableStates();
        const tableState = response.items.find((item: TableStateApiResponse) => item.id === table.id);
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

  const date = dayjs(lastUpdated);
  const shortDate = date.format('MMM D, YYYY');
  const fullDate = date.format('YYYY-MM-DD hh:mm A z');

  return (
    <Tooltip label={fullDate}>
      <Text size="sm" c="dimmed">
        Updated: {shortDate}
      </Text>
    </Tooltip>
  );
} 
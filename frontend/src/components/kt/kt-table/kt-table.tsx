import React, { useEffect, useState } from 'react';
import { Table, Text, ActionIcon, Box, Tooltip, Loader, Group } from '@mantine/core';
import { IconChevronDown, IconChevronUp, IconX, IconCheck, IconEdit, IconTrash } from '@tabler/icons-react';
import { notifications } from '@mantine/notifications';
import styled from '@emotion/styled';
import { useStore } from '../../../config/store/store';
import KtTableMenu from './kt-table-menu';

// Styled components
const TableContainer = styled.div`
  overflow-x: auto;
  width: 100%;
`;

const StyledTable = styled(Table)`
  width: 100%;
  min-width: 800px;
`;

const TableHeader = styled.th<{ sortable?: boolean }>`
  cursor: ${props => (props.sortable ? 'pointer' : 'default')};
  white-space: nowrap;
  position: relative;
  
  &:hover {
    background-color: ${props => props.sortable ? 'rgba(0, 0, 0, 0.05)' : 'transparent'};
  }
`;

const HeaderContent = styled.div`
  display: flex;
  align-items: center;
  gap: 5px;
`;

const TableCell = styled.td`
  max-width: 300px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const RowHighlight = styled.div<{ active?: boolean }>`
  background-color: ${props => props.active ? 'rgba(0, 120, 255, 0.1)' : 'transparent'};
  transition: background-color 0.2s;
`;

const HighlightSpan = styled.span<{ highlight?: boolean }>`
  background-color: ${props => props.highlight ? 'rgba(255, 230, 0, 0.3)' : 'transparent'};
  padding: ${props => props.highlight ? '0 2px' : '0'};
  border-radius: ${props => props.highlight ? '2px' : '0'};
`;

const ProcessingCell = styled.div`
  display: flex;
  align-items: center;
  color: #777;
`;

// Render a cell based on its content
const renderCell = (cell: any, row: KtRow) => {
  // Check if the cell has processing status
  if (cell && cell.result && cell.result.processing) {
    return (
      <ProcessingCell>
        <Loader size="xs" color="gray" />
        <Text size="xs" c="dimmed" ml={5}>Processing...</Text>
      </ProcessingCell>
    );
  }

  // Regular cell rendering logic
  if (typeof cell === 'boolean') {
    return cell ? <IconCheck color="green" /> : <IconX color="red" />;
  } else if (typeof cell === 'number') {
    return cell;
  } else if (Array.isArray(cell)) {
    // Display arrays as comma-separated list with item count
    return (
      <Tooltip label={cell.join(', ')}>
        <Text truncate>
          {cell.length > 0 ? `${cell.length} items: ${cell.join(', ')}` : '(empty)'}
        </Text>
      </Tooltip>
    );
  } else if (cell === null || cell === undefined) {
    return <Text c="dimmed">(empty)</Text>;
  } else if (typeof cell === 'object') {
    // For objects that have an answer property (from query results)
    if ('answer' in cell || 'result' in cell) {
      const value = cell.answer || (cell.result ? cell.result.answer : null);
      
      if (value === null || value === undefined) {
        return <Text c="dimmed">(no value)</Text>;
      } else if (typeof value === 'boolean') {
        return value ? <IconCheck color="green" /> : <IconX color="red" />;
      } else if (Array.isArray(value)) {
        return (
          <Tooltip label={value.join(', ')}>
            <Text truncate>
              {value.length > 0 ? `${value.length} items: ${value.join(', ')}` : '(empty)'}
            </Text>
          </Tooltip>
        );
      } else {
        return <HighlightedCell value={String(value)} row={row} />;
      }
    } else {
      // For other object types
      return <Text truncate>{JSON.stringify(cell)}</Text>;
    }
  } else {
    return <HighlightedCell value={String(cell)} row={row} />;
  }
}; 
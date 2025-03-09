import { Cell, CellTemplate, Compatible, Uncertain } from "@silevis/reactgrid";
import { useMutation } from "@tanstack/react-query";
import {
  Group,
  Text,
  ActionIcon,
  Tooltip,
  Divider,
  List,
  Button,
  FileButton,
  Loader
} from "@mantine/core";
import {
  IconFileText,
  IconPlus,
  IconRefresh,
  IconTrash,
  IconUpload,
  IconEye
} from "@tabler/icons-react";
import { CellPopover } from "./index.utils";
import { AnswerTableRow, useStore } from "@config/store";
import { useState } from "react";
import { KtDocumentPreview } from "../../kt-document-preview";

export interface KtRowCell extends Cell {
  type: "kt-row";
  row: AnswerTableRow;
}

export class KtRowCellTemplate implements CellTemplate<KtRowCell> {
  getCompatibleCell(cell: Uncertain<KtRowCell>): Compatible<KtRowCell> {
    if (cell.type !== "kt-row" || !cell.row) {
      throw new Error("Invalid cell type");
    }
    return {
      ...cell,
      type: "kt-row",
      row: cell.row,
      text: cell.row.sourceData?.type === "document" 
        ? cell.row.sourceData.document.name 
        : cell.row.sourceData?.type === "loading"
          ? `Loading: ${cell.row.sourceData.name}`
          : cell.row.sourceData?.type === "error"
            ? `Error: ${cell.row.sourceData.name}`
            : "",
      value: NaN
    };
  }

  isFocusable() {
    return false;
  }

  render({ row }: Compatible<KtRowCell>) {
    return <Content row={row} />;
  }
}

function Content({ row }: { row: AnswerTableRow }) {
  const [previewOpen, setPreviewOpen] = useState(false);
  
  const { mutateAsync: handleFillRow, isPending: isFillingRow } = useMutation({
    mutationFn: ({ id, file }: { id: string; file: File }) =>
      useStore.getState().fillRow(id, file)
  });

  const handleRerun = () => {
    useStore.getState().rerunRows([row.id]);
  };

  const handleDelete = () => {
    useStore.getState().deleteRows([row.id]);
  };
  
  const handleOpenPreview = () => {
    setPreviewOpen(true);
  };
  
  const handleClosePreview = () => {
    setPreviewOpen(false);
  };

  if (!row.sourceData) {
    return (
      <CellPopover
        monoClick
        target={
          <Group h="100%" pl="xs" gap="xs" wrap="nowrap">
            {isFillingRow ? (
              <Loader size="xs" />
            ) : (
              <IconPlus size={18} opacity={0.4} />
            )}
            <Text c="dimmed">Add data</Text>
          </Group>
        }
        dropdown={
          <>
            <FileButton
              accept="application/pdf,text/plain"
              onChange={file => file && handleFillRow({ id: row.id, file })}
            >
              {fileProps => (
                <Button
                  {...fileProps}
                  leftSection={
                    isFillingRow ? <Loader size="xs" /> : <IconUpload />
                  }
                >
                  Upload document
                </Button>
              )}
            </FileButton>
            <Text mt="xs" size="xs" c="dimmed">
              Accepted formats: pdf, txt
            </Text>
          </>
        }
      />
    );
  }

  // Handle different source data types
  if (row.sourceData.type === "loading") {
    return (
      <Group h="100%" pl="xs" gap="xs" wrap="nowrap">
        <Loader size="xs" />
        <Text>Loading: {row.sourceData.name}</Text>
      </Group>
    );
  }
  
  if (row.sourceData.type === "error") {
    return (
      <CellPopover
        target={
          <Group h="100%" pl="xs" gap="xs" wrap="nowrap" style={{ minWidth: 0 }}>
            <IconFileText size={18} color="red" style={{ flexShrink: 0 }} />
            <Text fw={500} truncate c="red">Error: {row.sourceData.name}</Text>
          </Group>
        }
        dropdown={
          <>
            <Group gap="xs" justify="space-between">
              <Text fw={500} c="red">Error: {row.sourceData.name}</Text>
              <Group gap="xs">
                <Tooltip label="Delete row">
                  <ActionIcon color="red" onClick={handleDelete}>
                    <IconTrash />
                  </ActionIcon>
                </Tooltip>
              </Group>
            </Group>
            <Divider mt="xs" mx="calc(var(--mantine-spacing-sm) * -1)" />
            <Text mt="xs" c="red">{row.sourceData.error}</Text>
            <div style={{ marginTop: 'var(--mantine-spacing-md)' }}>
              <FileButton
                accept="application/pdf,text/plain"
                onChange={file => file && handleFillRow({ id: row.id, file })}
              >
                {fileProps => (
                  <Button
                    {...fileProps}
                    fullWidth
                    leftSection={
                      isFillingRow ? <Loader size="xs" /> : <IconUpload />
                    }
                  >
                    Try again
                  </Button>
                )}
              </FileButton>
            </div>
          </>
        }
      />
    );
  }
  
  if (row.sourceData.type === "document") {
    return (
      <>
        <CellPopover
          target={
            <Group h="100%" pl="xs" gap="xs" wrap="nowrap" style={{ minWidth: 0 }}>
              <IconFileText size={18} opacity={0.7} style={{ flexShrink: 0 }} />
              <Text fw={500} truncate>{row.sourceData.document.name}</Text>
            </Group>
          }
          dropdown={
            <>
              <Group gap="xs" justify="space-between">
                <Text fw={500}>{row.sourceData.document.name}</Text>
                <Group gap="xs">
                  <Tooltip label="Preview document">
                    <ActionIcon color="blue" onClick={handleOpenPreview}>
                      <IconEye />
                    </ActionIcon>
                  </Tooltip>
                  <Tooltip label="Rerun row">
                    <ActionIcon onClick={handleRerun}>
                      <IconRefresh />
                    </ActionIcon>
                  </Tooltip>
                  <Tooltip label="Delete row">
                    <ActionIcon color="red" onClick={handleDelete}>
                      <IconTrash />
                    </ActionIcon>
                  </Tooltip>
                </Group>
              </Group>
              <Divider mt="xs" mx="calc(var(--mantine-spacing-sm) * -1)" />
              <List mt="xs">
                <List.Item>
                  <b>Type</b>: Document
                </List.Item>
                {row.sourceData.document.tag && (
                  <List.Item>
                    <b>Tag</b>: {row.sourceData.document.tag}
                  </List.Item>
                )}
                <List.Item>
                  <b>Author</b>: {row.sourceData.document.author || 'Unknown'}
                </List.Item>
                <List.Item>
                  <b>Page count</b>: {row.sourceData.document.page_count || 'Unknown'}
                </List.Item>
              </List>
              <Button 
                mt="md" 
                fullWidth 
                leftSection={<IconEye size={16} />}
                onClick={handleOpenPreview}
              >
                Preview Document
              </Button>
            </>
          }
        />
        {previewOpen && (
          <KtDocumentPreview 
            row={row} 
            onClose={handleClosePreview} 
          />
        )}
      </>
    );
  }
  
  // Default case - should not happen but TypeScript requires it
  return (
    <Group h="100%" pl="xs" gap="xs" wrap="nowrap">
      <Text>Unknown source data type</Text>
    </Group>
  );
}

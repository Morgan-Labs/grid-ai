import { useMemo } from "react";
import { Blockquote, Modal, Stack, Text } from "@mantine/core";
import { isEmpty, pick, values, isString, isArray } from "lodash-es";
import { useStore } from "@config/store";

export function KtChunks() {
  const table = useStore(store => store.getTable());
  const allChunks = table.chunks;
  const openedChunks = table.openedChunks;
  
  // Get all the answers from the cells that are being viewed
  const answers = useMemo(() => {
    if (isEmpty(openedChunks)) return [];
    
    const result: string[] = [];
    
    // For each opened chunk, find the corresponding cell value
    openedChunks.forEach(cellKey => {
      const [rowId, columnId] = cellKey.split('-');
      const row = table.rows.find(r => r.id === rowId);
      if (!row) return;
      
      const cellValue = row.cells[columnId];
      if (cellValue === undefined || cellValue === null) return;
      
      // Convert the cell value to string(s)
      if (isString(cellValue)) {
        result.push(cellValue);
      } else if (isArray(cellValue)) {
        result.push(...cellValue.map(String));
      } else {
        result.push(String(cellValue));
      }
    });
    
    return result;
  }, [openedChunks, table.rows]);
  
  const chunks = useMemo(
    () => values(pick(allChunks, openedChunks)).flat(),
    [allChunks, openedChunks]
  );
  
  // Function to highlight answers in text
  const highlightAnswers = (text: string) => {
    if (isEmpty(answers)) return text;
    
    // Simple highlighting approach - split by answer and join with highlighted version
    let result = text;
    for (const answer of answers) {
      if (!answer) continue;
      
      // Use a simple replace to highlight the answer
      // This is a basic approach and might have limitations with overlapping answers
      result = result.replace(
        new RegExp(answer, 'gi'),
        match => `<mark style="background-color: #ffeb3b;">${match}</mark>`
      );
    }
    
    return result;
  };

  return (
    <Modal
      size="xl"
      title="Chunks"
      opened={!isEmpty(openedChunks)}
      onClose={() => useStore.getState().closeChunks()}
    >
      {isEmpty(chunks) ? (
        <Text>No chunks found for selected cells</Text>
      ) : (
        <Stack>
          {chunks.map((chunk, index) => (
            <Blockquote key={index}>
              <div dangerouslySetInnerHTML={{ __html: highlightAnswers(chunk.text || chunk.content) }} />
            </Blockquote>
          ))}
        </Stack>
      )}
    </Modal>
  );
}

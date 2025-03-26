import { Cell, CellTemplate, Compatible, Uncertain } from "@silevis/reactgrid";
import { Group, Text, ColorSwatch, ActionIcon, Tooltip } from "@mantine/core";
import { IconSettings } from "@tabler/icons-react";
import { KtColumnSettings } from "./kt-column-settings";
import { KtColumnAddButton } from "./kt-column-add-button";
import { AnswerTableColumn, useStore } from "@config/store";
import { entityColor } from "@utils/functions";
import { useRef, useState } from "react";
import { CellPopover } from "./index.utils";

export interface KtColumnCell extends Cell {
  type: "kt-column";
  column: AnswerTableColumn;
  columnIndex?: number;
}

export class KtColumnCellTemplate implements CellTemplate<KtColumnCell> {
  getCompatibleCell(cell: Uncertain<KtColumnCell>): Compatible<KtColumnCell> {
    if (cell.type !== "kt-column" || !cell.column) {
      throw new Error("Invalid cell type");
    }
    return {
      ...cell,
      type: "kt-column",
      column: cell.column,
      text: cell.column.entityType,
      value: NaN
    };
  }

  isFocusable() {
    return false;
  }

  render({ column, columnIndex }: Compatible<KtColumnCell>) {
    const dragRef = useRef<HTMLDivElement>(null);
    const [isHeaderHovered, setIsHeaderHovered] = useState(false);

    const handleDragStart = (e: React.DragEvent<HTMLDivElement>) => {
      if (!columnIndex) return;
      
      // Make sure we're not starting a drag from the add buttons
      const target = e.target as HTMLElement;
      if (target.closest('.column-add-button')) {
        e.preventDefault();
        e.stopPropagation();
        return;
      }
      
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', String(columnIndex));
      
      // Add a custom class to the element being dragged
      setTimeout(() => {
        if (dragRef.current) {
          dragRef.current.classList.add('dragging');
        }
      }, 0);
    };
    
    const handleDragEnd = () => {
      if (dragRef.current) {
        dragRef.current.classList.remove('dragging');
      }
    };
    
    const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      
      // Add a class to indicate drop target
      if (dragRef.current) {
        dragRef.current.classList.add('drop-target');
        
        // Get the bounding rectangle of the target element
        const rect = e.currentTarget.getBoundingClientRect();
        // Get the x position of the mouse
        const x = e.clientX;
        // Calculate the middle point of the element
        const middle = rect.left + rect.width / 2;
        
        // Add classes to indicate drop position (left or right)
        if (x < middle) {
          dragRef.current.classList.add('drop-left');
          dragRef.current.classList.remove('drop-right');
        } else {
          dragRef.current.classList.add('drop-right');
          dragRef.current.classList.remove('drop-left');
        }
      }
    };
    
    const handleDragLeave = () => {
      // Remove drop target classes
      if (dragRef.current) {
        dragRef.current.classList.remove('drop-target');
        dragRef.current.classList.remove('drop-left');
        dragRef.current.classList.remove('drop-right');
      }
    };
    
    const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      
      // Remove drop target classes
      if (dragRef.current) {
        dragRef.current.classList.remove('drop-target');
        dragRef.current.classList.remove('drop-left');
        dragRef.current.classList.remove('drop-right');
      }
      
      if (columnIndex === undefined) return;
      
      const sourceIndex = parseInt(e.dataTransfer.getData('text/plain'), 10);
      const targetIndex = columnIndex;
      
      // Get the bounding rectangle of the target element
      const rect = e.currentTarget.getBoundingClientRect();
      // Get the x position of the drop
      const x = e.clientX;
      // Calculate the middle point of the element
      const middle = rect.left + rect.width / 2;
      
      // If dropping on the left half of the target, insert before the target
      // If dropping on the right half, insert after the target
      let newTargetIndex = targetIndex;
      if (x < middle) {
        // Drop on left side - insert before
        newTargetIndex = targetIndex;
      } else {
        // Drop on right side - insert after
        newTargetIndex = targetIndex + 1;
      }
      
      // Don't do anything if dropping on the same position
      if (sourceIndex === newTargetIndex || sourceIndex + 1 === newTargetIndex) {
        return;
      }
      
      // Adjust the target index if we're moving from left to right
      // because the removal of the source item shifts the indices
      if (sourceIndex < newTargetIndex) {
        newTargetIndex--;
      }
      
      useStore.getState().reorderColumns(sourceIndex, newTargetIndex);
    };

    return (
      <CellPopover
        monoClick
        mainAxisOffset={0}
        target={({ handleOpen }: { handleOpen: () => void }) => (
          <div 
            style={{ position: 'relative', width: '100%', height: '100%' }}
            onMouseEnter={() => setIsHeaderHovered(true)}
            onMouseLeave={() => setIsHeaderHovered(false)}
          >
            {/* Add column buttons */}
            {columnIndex !== undefined && (
              <>
                <KtColumnAddButton 
                  columnId={column.id} 
                  position="left" 
                  isHeaderHovered={isHeaderHovered} 
                />
                <KtColumnAddButton 
                  columnId={column.id} 
                  position="right" 
                  isHeaderHovered={isHeaderHovered} 
                />
              </>
            )}
            
            <Group 
              h="100%" 
              pl="xs" 
              gap="xs" 
              wrap="nowrap"
              ref={dragRef}
              draggable={columnIndex !== undefined}
              onDragStart={handleDragStart}
              onDragEnd={handleDragEnd}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              style={{ cursor: columnIndex !== undefined ? 'grab' : 'default' }}
            >
              <Tooltip label={column.generate ? (column.llmModel ? `LLM: ${column.llmModel}` : "LLM-generated column") : undefined}>
                <div style={{ 
                  position: 'relative', 
                  width: '12px', 
                  height: '12px',
                  borderRadius: '4px'
                }}>
                  <ColorSwatch
                    size={12}
                    color={entityColor(column.entityType).fill}
                  />
                  {column.generate && (
                    <>
                      <div 
                        className="pulse-animation"
                        style={{
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          right: 0,
                          bottom: 0,
                          borderRadius: '4px',
                          animation: 'pulse 3s infinite ease-in-out',
                          background: 'radial-gradient(circle, rgba(255,255,255,0.6) 0%, rgba(255,255,255,0) 70%)',
                          pointerEvents: 'none',
                          mixBlendMode: 'soft-light',
                          boxShadow: '0 0 2px 1px rgba(255,255,255,0.2)'
                        }} 
                      />
                      <Text
                        size="7px"
                        fw={800}
                        style={{
                          position: 'absolute',
                          top: '50%',
                          left: '50%',
                          transform: 'translate(-50%, -50%)',
                          color: 'white',
                          textShadow: '0px 0px 2px rgba(0,0,0,0.7)',
                          letterSpacing: '-0.3px',
                          lineHeight: 1,
                          userSelect: 'none',
                          pointerEvents: 'none'
                        }}
                      >
                        AI
                      </Text>
                    </>
                  )}
                </div>
              </Tooltip>
              <Text fw={500}>{column.entityType}</Text>
              <ActionIcon 
                variant="subtle" 
                size="xs"
                color="blue"
                ml="auto"
                mr="xs"
                onClick={handleOpen} // Call handleOpen when gear icon is clicked
              >
                <IconSettings size={14} />
              </ActionIcon>
            </Group>
          </div>
        )}
        dropdown={
          <KtColumnSettings
            value={column}
            onChange={(value, run) => {
              useStore.getState().editColumn(column.id, value);
              if (run) {
                useStore.getState().rerunColumns([column.id]);
              }
            }}
            onRerun={() => useStore.getState().rerunColumns([column.id])}
            onUnwind={() => useStore.getState().unwindColumn(column.id)}
            onHide={() => useStore.getState().editColumn(column.id, { hidden: true })}
            onDelete={() => useStore.getState().deleteColumns([column.id])}
          />
        }
      />
    );
  }
}

import { useState } from "react";
import { ActionIcon, Tooltip } from "@mantine/core";
import { IconColumnInsertLeft, IconColumnInsertRight } from "@tabler/icons-react";
import { useStore } from "@config/store";

interface KtColumnAddButtonProps {
  columnId: string;
  position: 'left' | 'right';
  isHeaderHovered: boolean;
}

export function KtColumnAddButton({ columnId, position, isHeaderHovered }: KtColumnAddButtonProps) {
  const [isButtonHovered, setIsButtonHovered] = useState(false);
  const insertColumnBefore = useStore(state => state.insertColumnBefore);
  const insertColumnAfter = useStore(state => state.insertColumnAfter);

  const handleAddColumn = (e: React.MouseEvent) => {
    // Stop propagation to prevent triggering drag events
    e.stopPropagation();
    
    if (position === 'left') {
      insertColumnBefore(columnId);
    } else {
      insertColumnAfter(columnId);
    }
  };

  // Only show the button when the header is hovered
  const isVisible = isHeaderHovered || isButtonHovered;

  return (
    <div
      className={`column-add-button ${position}`}
      onMouseEnter={() => setIsButtonHovered(true)}
      onMouseLeave={() => setIsButtonHovered(false)}
      onClick={(e) => e.stopPropagation()}
      style={{
        position: 'absolute',
        top: 0,
        bottom: 0,
        [position]: position === 'left' ? -12 : -12,
        width: 24,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 100,
        opacity: isVisible ? 1 : 0, // Only visible on hover
        transition: 'opacity 0.2s ease',
        cursor: 'pointer',
        pointerEvents: isVisible ? 'auto' : 'none' // Only clickable when visible
      }}
    >
      {isVisible && (
        <Tooltip label={`Insert column ${position === 'left' ? 'before' : 'after'}`}>
          <ActionIcon
            size="md"
            variant="filled"
            color="blue"
            onClick={handleAddColumn}
            style={{
              borderRadius: '50%',
              boxShadow: '0 3px 6px rgba(0, 0, 0, 0.2)',
              border: '2px solid white',
              transform: 'scale(1.1)',
              transition: 'transform 0.2s ease, box-shadow 0.2s ease'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'scale(1.2)';
              e.currentTarget.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.3)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'scale(1.1)';
              e.currentTarget.style.boxShadow = '0 3px 6px rgba(0, 0, 0, 0.2)';
            }}
          >
            {position === 'left' 
              ? <IconColumnInsertLeft size={18} stroke={2.5} /> 
              : <IconColumnInsertRight size={18} stroke={2.5} />
            }
          </ActionIcon>
        </Tooltip>
      )}
    </div>
  );
}

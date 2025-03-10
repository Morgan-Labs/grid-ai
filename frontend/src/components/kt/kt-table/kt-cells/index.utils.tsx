import { ReactNode, useEffect, useId } from "react";
import { Box, Popover, ScrollArea } from "@mantine/core";
import { Wrap } from "@components";
import { cn } from "@utils/functions";
import { useStore } from "@config/store";
import classes from "./index.module.css";

// Add global handlers to manage popovers
// This is done once at the module level to avoid adding multiple listeners
let isGlobalHandlersAdded = false;
let mentionSuggestionActive = false; // Track if mention suggestion is active

// Expose a function to set the active state
export function setMentionSuggestionActive(active: boolean) {
  mentionSuggestionActive = active;
}

const setupGlobalHandlers = () => {
  if (isGlobalHandlersAdded) return;
  
  // Click handler to close popovers when clicking outside
  document.addEventListener('click', (e) => {
    // Don't process if the target is null
    if (!e.target) return;
    
    // Get the active popover element
    const store = useStore.getState();
    if (!store.activePopoverId) return;
    
    // If mention suggestion is active, don't close popover
    if (mentionSuggestionActive) {
      return;
    }
    
    // Check if the click is inside the active popover's dropdown or target
    const isInsidePopover = !!(e.target as Element).closest(`.${classes.dropdown}`) || 
                            !!(e.target as Element).closest(`.${classes.target}`);
    
    // Check for specific mention suggestion elements that should not close the popover
    const isInsideMentionSuggestions = (function() {
      const targetEl = e.target as Element;
      
      // Check if the click is directly on a mention suggestion item
      // or any of its parent elements up to the suggestions list
      let currentEl: Element | null = targetEl;
      while (currentEl) {
        // Check for specific class names seen in the HTML structure
        if (currentEl.classList) {
          // These are the specific classes used in the mentions suggestion list
          if (currentEl.classList.contains('mentions__suggestions__item') || 
              currentEl.classList.contains('mentions__suggestions__list') ||
              currentEl.classList.contains('_suggestion_dyb3w_38') ||
              currentEl.classList.contains('_active_dyb3w_44')) {
            return true;
          }
        }
        
        // Also check for specific role attributes
        const role = currentEl.getAttribute('role');
        if (role === 'option' || role === 'listbox') {
          return true;
        }
        
        // Look for specific parent elements that might contain mentions
        if (currentEl.tagName === 'UL' && currentEl.hasAttribute('id') && 
            currentEl.getAttribute('id')?.includes('mentions')) {
          return true;
        }
        
        // Move up to parent
        currentEl = currentEl.parentElement;
      }
      
      return false;
    })();
    
    // If click is inside the popover or mention suggestions, don't close it
    if (isInsidePopover || isInsideMentionSuggestions) {
      // console.log("Click inside popover or mention suggestions, not closing");
      return;
    }
    
    // If click is outside, close the popover
    store.setActivePopover(null);
  });
  
  // Keyboard handler to close popovers when Escape is pressed
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      const store = useStore.getState();
      if (store.activePopoverId) {
        store.setActivePopover(null);
      }
    }
  });
  
  isGlobalHandlersAdded = true;
};

// Setup the global handlers
if (typeof document !== 'undefined') {
  setupGlobalHandlers();
}

interface CellPopoverProps {
  monoClick?: boolean;
  mainAxisOffset?: number;
  target: ReactNode | ((props: { handleOpen: () => void }) => ReactNode);
  dropdown: ReactNode;
  scrollable?: boolean;
}

export { CellPopover };
export type { CellPopoverProps };

function CellPopover({
  monoClick,
  mainAxisOffset = 1,
  target,
  dropdown,
  scrollable
}: CellPopoverProps) {
  // Generate a unique ID for this popover
  const id = useId();
  
  // Use global state to track which popover is active
  const activePopoverId = useStore(state => state.activePopoverId);
  const setActivePopover = useStore(state => state.setActivePopover);
  
  // Determine if this popover is open
  const opened = activePopoverId === id;
  
  // Close popover when component unmounts
  useEffect(() => {
    return () => {
      if (activePopoverId === id) {
        setActivePopover(null);
      }
    };
  }, [activePopoverId, id, setActivePopover]);
  
  // Handle opening and closing
  const handleOpen = () => {
    setActivePopover(id);
  };
  
  const handleClose = () => {
    if (activePopoverId === id) {
      setActivePopover(null);
    }
  };
  
  return (
    <Popover
      opened={opened}
      onClose={handleClose}
      offset={{ mainAxis: mainAxisOffset, crossAxis: -1 }}
      width="target"
      position="bottom-start"
      transitionProps={{ transition: "scale-y" }}
      withinPortal={true}
    >
      <Popover.Target>
        <Box
          className={cn(classes.target, opened && classes.active)}
          {...(monoClick
            ? { onClick: handleOpen }
            : { onDoubleClick: handleOpen })}
        >
          {typeof target === 'function' ? target({ handleOpen }) : target}
        </Box>
      </Popover.Target>
      <Popover.Dropdown
        onPointerDown={e => e.stopPropagation()}
        onKeyDown={e => e.stopPropagation()}
        className={classes.dropdown}
      >
        <Wrap
          with={
            scrollable &&
            (node => (
              <ScrollArea.Autosize mah={500}>{node}</ScrollArea.Autosize>
            ))
          }
        >
          <Box p="sm">{dropdown}</Box>
        </Wrap>
      </Popover.Dropdown>
    </Popover>
  );
}

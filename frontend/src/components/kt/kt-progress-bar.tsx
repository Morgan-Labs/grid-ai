import { useEffect, useState, useRef } from 'react';
import { Progress, Text, Group, Box, Paper, ActionIcon } from '@mantine/core';
import { IconX } from '@tabler/icons-react';
import { useStore } from '@config/store';
import { notifications } from '@utils/notifications';

/**
 * Simple progress bar component that shows the current progress of batch operations
 */
export function KtProgressBar() {
  // Get progress data directly from store
  const requestProgress = useStore(store => store.getTable().requestProgress);
  const [dismissed, setDismissed] = useState(false);
  
  // Track previous state to detect transitions
  const prevInProgressRef = useRef<boolean | undefined>(undefined);
  
  // Reset dismissed state when a new operation starts
  useEffect(() => {
    if (requestProgress?.inProgress) {
      setDismissed(false);
    }
  }, [requestProgress?.inProgress]);
  
  // Show notification when processing completes
  useEffect(() => {
    if (!requestProgress) return;
    
    const wasInProgress = prevInProgressRef.current;
    const isNowComplete = !requestProgress.inProgress && requestProgress.completed > 0 && !requestProgress.error;
    
    // Update the ref for next time
    prevInProgressRef.current = requestProgress.inProgress;
    
    // Only show notification if we were previously in progress and now we're complete
    if (wasInProgress === true && isNowComplete) {
      notifications.show({
        title: 'Processing complete',
        message: `Successfully processed all ${requestProgress.completed} requests.`,
        color: 'green',
        autoClose: 5000
      });
    }
  }, [requestProgress?.inProgress, requestProgress?.completed, requestProgress?.error]);
  
  // Don't render if no progress data or dismissed
  if (!requestProgress || dismissed) {
    return null;
  }
  
  // Only show progress bar if there's actual progress to show
  if (requestProgress.total === 0) {
    return null;
  }
  
  // Calculate the actual values to display
  const total = requestProgress.total || 0;
  
  // Get the raw completed count
  const rawCompleted = requestProgress.completed || 0;
  
  // FIXED LOGIC: Only reset to 0 if we're at the very beginning of processing
  // This allows progress to update correctly during processing
  const completed = requestProgress.inProgress && rawCompleted === total && rawCompleted > 0
    ? Math.min(20, Math.floor(total * 0.05))  // Show small initial progress (5%) to indicate processing has started
    : Math.min(rawCompleted, total);  // Otherwise use the actual count (capped at total)
  
  // Calculate percentage
  const percentage = total > 0 
    ? Math.min(100, Math.max(0, Math.round((completed / total) * 100))) 
    : 0;
  
  // Determine the status message
  const statusMessage = !requestProgress.inProgress 
    ? (requestProgress.error ? "Processing failed" : "Processing complete") 
    : "Processing requests";
  
  // Calculate estimated time remaining (ETA)
  let etaText = null;
  if (requestProgress.inProgress && total > 0 && completed > 0) {
    // More accurate estimate based on batch processing
    // A batch of 20 queries takes about 3-5 seconds
    const batchSize = 20;
    const secondsPerBatch = 5; // Average time for a batch of 20 queries
    
    const remainingQueries = total - completed;
    const remainingBatches = Math.ceil(remainingQueries / batchSize);
    const estimatedSecondsRemaining = remainingBatches * secondsPerBatch;
    
    // Add a small buffer for network latency and processing overhead
    const adjustedSecondsRemaining = Math.ceil(estimatedSecondsRemaining * 1.1);
    
    if (adjustedSecondsRemaining < 60) {
      etaText = `${adjustedSecondsRemaining} seconds remaining`;
    } else if (adjustedSecondsRemaining < 3600) {
      etaText = `${Math.ceil(adjustedSecondsRemaining / 60)} minutes remaining`;
    } else {
      const hours = Math.floor(adjustedSecondsRemaining / 3600);
      const minutes = Math.ceil((adjustedSecondsRemaining % 3600) / 60);
      etaText = `${hours}h ${minutes}m remaining`;
    }
  } else if (requestProgress.inProgress && total > 0) {
    // If we haven't processed any queries yet, show a calculating message
    etaText = "Calculating...";
  }
  
  return (
    <Paper 
      shadow="md" 
      p="md" 
      withBorder 
      style={{ 
        position: 'fixed', 
        bottom: 80,
        right: 20, 
        zIndex: 999,
        width: 300,
        backgroundColor: 'white',
        borderLeft: `4px solid ${requestProgress.error ? '#fa5252' : '#228be6'}`,
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
        pointerEvents: 'auto'
      }}
    >
      <Box>
        <Group justify="space-between" mb={5}>
          <Text size="sm" fw={500}>{statusMessage}</Text>
          <Group gap="xs">
            <Text size="sm" c="dimmed">
              {completed} / {total}
            </Text>
            <ActionIcon 
              size="xs" 
              variant="subtle" 
              onClick={() => setDismissed(true)}
              aria-label="Close progress bar"
            >
              <IconX size={14} />
            </ActionIcon>
          </Group>
        </Group>
        <Progress 
          value={percentage} 
          size="md" 
          radius="xl" 
          color={requestProgress.error ? "red" : "blue"}
          striped={requestProgress.inProgress}
          animated={requestProgress.inProgress}
        />
        <Group justify="space-between" mt={5}>
          <Text size="xs" c="dimmed">
            {percentage}% complete
          </Text>
          {etaText && (
            <Text size="xs" c="dimmed">
              {etaText}
            </Text>
          )}
        </Group>
      </Box>
    </Paper>
  );
}

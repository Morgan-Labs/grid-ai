import { Button, Group, BoxProps } from "@mantine/core";
import { IconDownload, IconUpload } from "@tabler/icons-react";
import { useStore } from "@config/store";
import { notifications } from "@utils/notifications";
import { useState } from "react";

export function KtPersistence(props: BoxProps) {
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(false);
  
  const isAuthenticated = useStore(state => state.auth.isAuthenticated);
  
  const handleSaveState = async () => {
    setSaving(true);
    try {
      await useStore.getState().saveTableState();
      
      notifications.show({
        title: 'Table state saved',
        message: 'Your table state has been saved to the database',
        color: 'green'
      });
    } catch (error) {
      console.error('Error saving table state:', error);
      
      notifications.show({
        title: 'Error saving table state',
        message: error instanceof Error ? error.message : 'Unknown error',
        color: 'red'
      });
    } finally {
      setSaving(false);
    }
  };
  
  const handleLoadState = async () => {
    setLoading(true);
    try {
      await useStore.getState().loadLatestTableState();
      
      notifications.show({
        title: 'Table state loaded',
        message: 'The latest table state has been loaded successfully',
        color: 'green'
      });
    } catch (error) {
      console.error('Error loading table state:', error);
      
      notifications.show({
        title: 'Error loading table state',
        message: error instanceof Error ? error.message : 'Unknown error',
        color: 'red'
      });
    } finally {
      setLoading(false);
    }
  };
  
  if (!isAuthenticated) {
    return null; // Don't show the component if not authenticated
  }
  
  return (
    <Group gap={8} {...props}>
      <Button 
        leftSection={<IconDownload />} 
        onClick={handleSaveState}
        loading={saving}
        variant="light"
        title="Save table state to database"
      >
        Save
      </Button>
      
      <Button 
        leftSection={<IconUpload />} 
        onClick={handleLoadState}
        loading={loading}
        variant="light"
        title="Load latest table state from database"
      >
        Load
      </Button>
    </Group>
  );
}

import { QueryClientProvider } from "@tanstack/react-query";
import { useEffect } from "react";
import {
  ActionIcon,
  Divider,
  Group,
  MantineProvider,
  Paper,
  Text,
  Tooltip
} from "@mantine/core";
import { ModalsProvider } from "@mantine/modals";
import {
  IconDatabase,
  IconMoon,
  IconSun
} from "@tabler/icons-react";
import "@mantine/core/styles.css";
import "@mantine/dropzone/styles.css";
import "@silevis/reactgrid/styles.css";
import { queryClient } from "@config/query";
import { useTheme } from "@config/theme";
import { useStore } from "@config/store";
import { KtTable, KTFileDrop, KtSwitch, KtControls } from "@components";
import { AuthWrapper } from "./components/auth";
import { KtAutoPersistence } from "./components/kt/kt-auto-persistence";
import { updateDocumentStatusesInTableState } from "./utils/updateDocumentStatuses";
import "./app.css";

export function App() {
  const theme = useTheme();
  const colorScheme = useStore(store => store.colorScheme);
  const isAuthenticated = useStore(store => store.auth.isAuthenticated);
  
  // Update document statuses when the app initializes and the user is authenticated
  useEffect(() => {
    if (isAuthenticated) {
      // Wait for the table state to be loaded first
      const loadAndUpdate = async () => {
        try {
          // First, load the latest table state
          await useStore.getState().loadLatestTableState();
          
          // Add a small delay to ensure the table state is fully loaded
          await new Promise(resolve => setTimeout(resolve, 500));
          
          // Then update document statuses - prioritize backend status over table state
          console.log("Checking document statuses from backend API...");
          await updateDocumentStatusesInTableState();
          console.log("Document status check completed");
        } catch (error) {
          console.error("Error updating document statuses:", error);
        }
      };
      
      loadAndUpdate();
    }
  }, [isAuthenticated]);
  
  return (
    <QueryClientProvider client={queryClient}>
      <MantineProvider theme={theme} forceColorScheme={colorScheme}>
        <ModalsProvider>
          <AuthWrapper>
            {/* Main Header */}
            <Paper 
              shadow="xs" 
              p="md" 
              className="app-header"
              style={{
                position: 'sticky',
                top: 0,
                zIndex: 100,
                borderRadius: 0,
                backgroundColor: colorScheme === 'dark' ? 
                  'var(--mantine-color-dark-7)' : 
                  'var(--mantine-color-gray-0)'
              }}
            >
              <Group justify="space-between" align="center">
                {/* Logo and Table Selector */}
                <Group>
                  <Group gap="xs" style={{ alignItems: "center" }}>
                    <IconDatabase size={24} style={{ color: 'var(--mantine-color-blue-6)' }} />
                    <Text fw={700} size="lg">AI Grid</Text>
                  </Group>
                  <Divider orientation="vertical" />
                  <KtSwitch />
                </Group>
                
                {/* Right side controls */}
                <Tooltip label={`Switch to ${colorScheme === 'light' ? 'dark' : 'light'} mode`}>
                  <ActionIcon 
                    variant="subtle" 
                    onClick={useStore.getState().toggleColorScheme}
                    size="md"
                  >
                    {colorScheme === "light" ? <IconMoon size={18} /> : <IconSun size={18} />}
                  </ActionIcon>
                </Tooltip>
              </Group>
            </Paper>
            
            {/* Controls and Table */}
            <KtControls className="excel-controls" />
            <KtTable />
            
            <KTFileDrop />
            <KtAutoPersistence />
          </AuthWrapper>
        </ModalsProvider>
      </MantineProvider>
    </QueryClientProvider>
  );
}

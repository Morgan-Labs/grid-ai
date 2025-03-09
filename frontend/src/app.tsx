import { QueryClientProvider } from "@tanstack/react-query";
import { ActionIcon, Divider, Group, MantineProvider } from "@mantine/core";
import { ModalsProvider } from "@mantine/modals";
import { IconMoon, IconSun } from "@tabler/icons-react";
import "@mantine/core/styles.css";
import "@mantine/dropzone/styles.css";
import "@silevis/reactgrid/styles.css";
import { queryClient } from "@config/query";
import { useTheme } from "@config/theme";
import { useStore } from "@config/store";
import { KtTable, KTFileDrop, KtSwitch, KtControls } from "@components";
import { AuthWrapper } from "./components/auth";
import { KtAutoPersistence } from "./components/kt/kt-auto-persistence";
import "./app.css";

export function App() {
  const theme = useTheme();
  const colorScheme = useStore(store => store.colorScheme);
  
  return (
    <QueryClientProvider client={queryClient}>
      <MantineProvider theme={theme} forceColorScheme={colorScheme}>
        <ModalsProvider>
          <AuthWrapper>
            <Group p="md" justify="space-between">
              <Group>
                <KtSwitch />
                {/* Test Errors button commented out
                <Button 
                  variant="subtle" 
                  leftSection={<IconBug size={16} />}
                  onClick={() => setErrorTestOpen(true)}
                >
                  Test Errors
                </Button>
                */}
              </Group>
              <ActionIcon onClick={useStore.getState().toggleColorScheme}>
                {colorScheme === "light" ? <IconSun /> : <IconMoon />}
              </ActionIcon>
            </Group>
            
            {/* Error Test Modal - commented out
            <Modal
              opened={errorTestOpen}
              onClose={() => setErrorTestOpen(false)}
              title="Error Testing"
              size="lg"
            >
              <ErrorTest />
            </Modal>
            */}
            <Divider />
            <KtControls mt="md" ml="md" />
            <KtTable mt="md" />
            <KTFileDrop />
            <KtAutoPersistence />
          </AuthWrapper>
        </ModalsProvider>
      </MantineProvider>
    </QueryClientProvider>
  );
}

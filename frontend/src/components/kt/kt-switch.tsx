import {
  Text,
  BoxProps,
  Button,
  Group,
  Menu,
  TextInput,
  ActionIcon,
  Tooltip,
  Badge,
  Stack
} from "@mantine/core";
import { modals } from "@mantine/modals";
import { useInputState } from "@mantine/hooks";
import {
  IconChevronDown,
  IconDeviceFloppy,
  IconPencil,
  IconPlus,
  IconTable,
  IconTrash,
  IconWand,
  IconClock
} from "@tabler/icons-react";
import { AnswerTable, useStore, TableStateListItem } from "@config/store";
import { useDerivedState } from "@hooks";
import { useState } from "react";

export function KtSwitch(props: BoxProps) {
  const activeTable = useStore(store => store.getTable());
  const savedStates = useStore(store => store.savedStates || []);
  const loadTableState = useStore(store => store.loadTableState);
  const allTablesFromStore = useStore(store => store.tables);
  const [sortByUpdated, setSortByUpdated] = useState(true);

  const handleNewTable = () => {
    modals.open({
      size: "xs",
      title: "New table",
      children: <NewTableModalContent />
    });
  };

  const handleRename = () => {
    if (!activeTable) return;
    modals.open({
      size: "xs",
      title: "Rename table",
      children: <RenameTableModalContent table={activeTable} />
    });
  };

  const handleDelete = () => {
    if (!activeTable) return;
    modals.openConfirmModal({
      title: "Delete table",
      children: (
        <Text>
          Are you sure you want to delete table '{activeTable.name}'?
        </Text>
      ),
      labels: { confirm: "Confirm", cancel: "Cancel" },
      onConfirm: () => useStore.getState().deleteTable(activeTable.id)
    });
  };

  const formatTimestamp = (isoString: string) => {
    if (!isoString) return null;
    try {
        const date = new Date(isoString + 'Z');
        return {
          short: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
          full: date.toLocaleString('en-US', { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', timeZoneName: 'short' })
        };
    } catch (e) {
        console.error("Error parsing date:", isoString, e);
        return null;
    }
  };

  const sortedSavedStates = [...savedStates].sort((a, b) => {
    if (sortByUpdated) {
      return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
    } else {
      return a.name.localeCompare(b.name);
    }
  });

  return (
    <Group gap="xs" {...props}>
      <Group gap="xs">
        <Menu position="bottom-start" shadow="md">
          <Menu.Target>
            <Button 
              variant="light" 
              rightSection={<IconChevronDown size={16} />}
              leftSection={<IconTable size={16} />}
              disabled={!activeTable}
            >
              {activeTable ? activeTable.name : "Loading..."}
            </Button>
          </Menu.Target>
          <Menu.Dropdown>
            <Menu.Label>
              <Group justify="space-between">
                <Text>Tables</Text>
                <Tooltip label={sortByUpdated ? "Sort by name" : "Sort by last updated"}>
                  <ActionIcon variant="subtle" onClick={() => setSortByUpdated(!sortByUpdated)} color={sortByUpdated ? "blue" : "gray"}>
                    <IconClock size={16} />
                  </ActionIcon>
                </Tooltip>
              </Group>
            </Menu.Label>
            {sortedSavedStates.map(stateItem => {
              const lastUpdated = formatTimestamp(stateItem.updated_at);
              const isActive = activeTable?.id === stateItem.id;
              return (
                <Menu.Item
                  key={stateItem.id}
                  leftSection={<IconTable size={16} />}
                  onClick={() => !isActive && loadTableState(stateItem.id)}
                  fw={isActive ? 700 : 'normal'}
                  rightSection={
                    <Stack gap={0} align="end">
                      {isActive && <Badge size="xs" variant="light">Active</Badge>}
                      {lastUpdated && (
                        <Tooltip label={lastUpdated.full}>
                          <Text size="xs" c="dimmed">
                            {lastUpdated.short}
                          </Text>
                        </Tooltip>
                      )}
                    </Stack>
                  }
                >
                  {stateItem.name}
                </Menu.Item>
              );
            })}
            <Menu.Divider />
            <Menu.Item 
              leftSection={<IconPlus size={16} />} 
              onClick={handleNewTable}
              color="blue"
            >
              New table
            </Menu.Item>
          </Menu.Dropdown>
        </Menu>
        
        <Group gap={4}>
          <Tooltip label="Rename table">
            <ActionIcon variant="subtle" onClick={handleRename} size="md" disabled={!activeTable}>
              <IconPencil size={16} />
            </ActionIcon>
          </Tooltip>
          {allTablesFromStore.length > 1 && (
            <Tooltip label="Delete table">
              <ActionIcon variant="subtle" color="red" onClick={handleDelete} size="md" disabled={!activeTable}>
                <IconTrash size={16} />
              </ActionIcon>
            </Tooltip>
          )}
        </Group>
      </Group>
    </Group>
  );
}

function NewTableModalContent() {
  const [name, setName] = useInputState("");
  const [error, setError] = useInputState("");
  const tables = useStore(store => store.tables);
  
  const handleCreate = () => {
    const nameExists = tables.some(t => t.name.toLowerCase() === name.trim().toLowerCase());
    
    if (nameExists) {
      setError("A table with this name already exists. Please choose a different name.");
      return;
    }
    
    useStore.getState().addTable(name.trim());
    modals.closeAll();
  };

  const handleNameChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setName(event.currentTarget.value);
    if (error) setError("");
  };

  return (
    <>
      <TextInput
        autoFocus
        label="Name"
        placeholder="Table name"
        value={name}
        onChange={handleNameChange}
        onKeyDown={e => !e.ctrlKey && e.key === "Enter" && handleCreate()}
        error={error}
      />
      <Button
        mt="md"
        fullWidth
        color="blue"
        disabled={!name.trim()}
        onClick={handleCreate}
        leftSection={<IconWand />}
      >
        Create
      </Button>
    </>
  );
}

function RenameTableModalContent({ table }: { table: AnswerTable }) {
  const [name, handlers] = useDerivedState(table.name);
  const [error, setError] = useInputState("");
  const tables = useStore(store => store.tables);
  
  const handleSave = () => {
    const nameExists = tables.some(t => 
      t.id !== table.id && 
      t.name.toLowerCase() === name.trim().toLowerCase()
    );
    
    if (nameExists) {
      setError("A table with this name already exists. Please choose a different name.");
      return;
    }
    
    useStore.getState().editTable(table.id, { name: name.trim() });
    modals.closeAll();
  };
  
  const handleNameChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    handlers.set(event.currentTarget.value);
    if (error) setError("");
  };

  return (
    <>
      <TextInput
        autoFocus
        label="Name"
        placeholder="Table name"
        value={name}
        onChange={handleNameChange}
        onKeyDown={e => !e.ctrlKey && e.key === "Enter" && handleSave()}
        error={error}
      />
      <Button
        mt="md"
        fullWidth
        color="blue"
        disabled={!handlers.dirty || !name.trim()}
        onClick={handleSave}
        leftSection={<IconDeviceFloppy />}
      >
        Save
      </Button>
    </>
  );
}

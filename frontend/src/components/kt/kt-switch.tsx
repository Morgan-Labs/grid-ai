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
import { AnswerTable, useStore } from "@config/store";
import { useDerivedState } from "@hooks";
import { listTableStates, TableState } from "../../services/api/table-state";
import { useState, useEffect } from "react";

export function KtSwitch(props: BoxProps) {
  const table = useStore(store => store.getTable());
  const tables = useStore(store => store.tables);
  const [tableStates, setTableStates] = useState<TableState[]>([]);
  const [sortByUpdated, setSortByUpdated] = useState(true);

  useEffect(() => {
    const fetchTableStates = async () => {
      try {
        const response = await listTableStates();
        setTableStates(response.items);
      } catch (error) {
        console.error('Failed to fetch table states:', error);
      }
    };
    fetchTableStates();
  }, []);

  const handleNewTable = () => {
    modals.open({
      size: "xs",
      title: "New table",
      children: <NewTableModalContent />
    });
  };

  const handleRename = () => {
    modals.open({
      size: "xs",
      title: "Rename table",
      children: <RenameTableModalContent table={table} />
    });
  };

  const handleDelete = () => {
    modals.openConfirmModal({
      title: "Delete table",
      children: (
        <Text>
          Are you sure you want to delete this table and all its data?
        </Text>
      ),
      labels: { confirm: "Confirm", cancel: "Cancel" },
      onConfirm: () => useStore.getState().deleteTable(table.id)
    });
  };

  const getLastUpdated = (tableId: string) => {
    const tableState = tableStates.find(state => state.id === tableId);
    if (!tableState) return null;
    const date = new Date(tableState.updated_at + 'Z');
    return {
      short: date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
      }),
      full: date.toLocaleString('en-US', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        timeZoneName: 'short'
      })
    };
  };

  const sortedTables = [...tables].sort((a, b) => {
    if (sortByUpdated) {
      const aUpdated = tableStates.find(state => state.id === a.id)?.updated_at;
      const bUpdated = tableStates.find(state => state.id === b.id)?.updated_at;
      if (!aUpdated || !bUpdated) return 0;
      return new Date(bUpdated).getTime() - new Date(aUpdated).getTime();
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
            >
              {table.name}
            </Button>
          </Menu.Target>
          <Menu.Dropdown>
            <Menu.Label>
              <Group justify="space-between">
                <Text>Tables</Text>
                <Tooltip label={sortByUpdated ? "Sort by name" : "Sort by last updated"}>
                  <ActionIcon 
                    variant="subtle" 
                    onClick={() => setSortByUpdated(!sortByUpdated)}
                    color={sortByUpdated ? "blue" : "gray"}
                  >
                    <IconClock size={16} />
                  </ActionIcon>
                </Tooltip>
              </Group>
            </Menu.Label>
            {sortedTables.map(t => {
              const lastUpdated = getLastUpdated(t.id);
              return (
                <Menu.Item
                  key={t.id}
                  leftSection={<IconTable size={16} />}
                  onClick={() => useStore.getState().switchTable(t.id)}
                  rightSection={
                    <Stack gap={0} align="end">
                      {t.id === table.id && <Badge size="xs" variant="light">Active</Badge>}
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
                  {t.name}
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
            <ActionIcon variant="subtle" onClick={handleRename} size="md">
              <IconPencil size={16} />
            </ActionIcon>
          </Tooltip>
          {tables.length > 1 && (
            <Tooltip label="Delete table">
              <ActionIcon variant="subtle" color="red" onClick={handleDelete} size="md">
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
    // Check if a table with this name already exists
    const nameExists = tables.some(t => t.name.toLowerCase() === name.trim().toLowerCase());
    
    if (nameExists) {
      setError("A table with this name already exists. Please choose a different name.");
      return;
    }
    
    useStore.getState().addTable(name.trim());
    modals.closeAll();
  };

  // Clear error when name changes
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
    // Check if a table with this name already exists (excluding the current table)
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
  
  // Clear error when name changes
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

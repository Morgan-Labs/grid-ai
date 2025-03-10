import {
  Text,
  Box,
  BoxProps,
  Button,
  Group,
  Menu,
  TextInput,
  ActionIcon,
  Tooltip
} from "@mantine/core";
import { modals } from "@mantine/modals";
import { useInputState } from "@mantine/hooks";
import {
  IconChevronDown,
  IconDatabase,
  IconDeviceFloppy,
  IconPencil,
  IconPlus,
  IconTable,
  IconTrash,
  IconWand
} from "@tabler/icons-react";
import { AnswerTable, useStore } from "@config/store";
import { useDerivedState } from "@hooks";

export function KtSwitch(props: BoxProps) {
  const table = useStore(store => store.getTable());
  const tables = useStore(store => store.tables);

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

  return (
    <Group gap="xs" {...props}>
<Box style={{ display: "flex", alignItems: "center", gap: "8px" }}>
  <IconDatabase size={20} />
  <Text fw={700} style={{ marginLeft: 4 }}>AI Grid</Text>
</Box>
      <Menu>
        <Menu.Target>
          <Button rightSection={<IconChevronDown />}>{table.name}</Button>
        </Menu.Target>
        <Menu.Dropdown>
          {tables.map(t => (
            <Menu.Item
              key={t.id}
              leftSection={<IconTable />}
              onClick={() => useStore.getState().switchTable(t.id)}
            >
              {t.name}
            </Menu.Item>
          ))}
          <Menu.Item leftSection={<IconPlus />} onClick={handleNewTable}>
            New table
          </Menu.Item>
        </Menu.Dropdown>
      </Menu>
      <Tooltip label="Rename table">
        <ActionIcon onClick={handleRename}>
          <IconPencil />
        </ActionIcon>
      </Tooltip>
      {tables.length > 1 && (
        <Tooltip label="Delete table">
          <ActionIcon color="red" onClick={handleDelete}>
            <IconTrash />
          </ActionIcon>
        </Tooltip>
      )}
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

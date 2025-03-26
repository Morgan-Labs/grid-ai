import { BoxProps, Button, Group, Text, Loader } from "@mantine/core";
import { IconEyeOff } from "@tabler/icons-react";
import { KtHiddenPill } from "./kt-hidden-pill";
import { KtClear } from "./kt-clear";
import { KtFilters } from "./kt-filters";
import { KTGlobalRules } from "./kt-global-rules";
import { KtResolvedEntities } from "./kt-resolved-entities";
import { KtDownload } from "./kt-download";
import { KtCsvUpload } from "../kt-csv-upload";
import { KtChunks } from "./kt-chunks";
import { useStore } from "@config/store";

export function KtControls(props: BoxProps) {
  const uploadingFiles = useStore(store => store.getTable().uploadingFiles);

  return (
    <Group className="excel-ribbon" p="xs" {...props}>
      <KtCsvUpload />
      <KtDownload.Csv />
      <KtClear />
      <Button
        leftSection={<IconEyeOff />}
        onClick={() => useStore.getState().toggleAllColumns(true)}
      >
        Hide all columns
      </Button>
      <KtHiddenPill />
      <KtFilters />
      <KTGlobalRules />
      <KtResolvedEntities />
      <KtChunks />
      
      {uploadingFiles && (
        <Group>
          <Loader size="xs" />
          <Text>Uploading files...</Text>
        </Group>
      )}
    </Group>
  );
}

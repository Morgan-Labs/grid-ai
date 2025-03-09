import { useEffect, useState, useRef } from "react";
import { 
  Box, 
  Button, 
  Card, 
  Drawer, 
  Group, 
  Loader, 
  ScrollArea, 
  Stack, 
  Text, 
  Title 
} from "@mantine/core";
import { IconFileText, IconX } from "@tabler/icons-react";
import { useStore } from "@config/store";
import { AnswerTableRow } from "@config/store/store.types";
import { fetchDocumentPreview } from "@config/api";

interface DocumentPreviewProps {
  row: AnswerTableRow | null;
  onClose: () => void;
}

export function KtDocumentPreview({ row, onClose }: DocumentPreviewProps) {
  const [loading, setLoading] = useState(false);
  const [content, setContent] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  
  // Use refs to track state between renders
  const contentFetchedRef = useRef(false);
  const documentIdRef = useRef<string | null>(null);
  
  // Get document from row
  const document = row?.sourceData?.type === 'document' ? row.sourceData.document : null;
  
  // Get store data
  const { documentPreviews, addDocumentPreview } = useStore(state => ({
    documentPreviews: state.documentPreviews,
    addDocumentPreview: state.addDocumentPreview
  }));
  
  // Get chunks from the store for this row
  const chunks = useStore(state => {
    if (!row) return [];
    
    // Collect all chunks for this row
    const allChunks = state.getTable().chunks;
    const rowChunks = Object.entries(allChunks)
      .filter(([key]) => key.startsWith(`${row.id}-`))
      .flatMap(([_, chunks]) => chunks);
      
    return rowChunks;
  });
  
  // Load document content
  useEffect(() => {
    // Skip if no document
    if (!document) {
      setContent([]);
      setError(null);
      contentFetchedRef.current = false;
      documentIdRef.current = null;
      return;
    }
    
    // Skip if document hasn't changed
    if (document.id === documentIdRef.current && contentFetchedRef.current) {
      return;
    }
    
    // Update document ID ref
    documentIdRef.current = document.id;
    
    // If we have chunks, use them as the document content
    if (chunks.length > 0) {
      setContent(chunks.map(chunk => chunk.text || chunk.content));
      setError(null);
      contentFetchedRef.current = true;
      return;
    }
    
    // Check if we already have the document preview in the store
    if (documentPreviews[document.id]) {
      // Avoid logging here to prevent console spam
      setContent(documentPreviews[document.id]);
      setError(null);
      contentFetchedRef.current = true;
      return;
    }
    
    // Otherwise, fetch the document content from the preview endpoint
    setLoading(true);
    setError(null);
    contentFetchedRef.current = true;
    
    // Define the fetch function inside the effect to avoid dependency issues
    const fetchContent = async () => {
      try {
        // Fetch document content using the preview endpoint
        const documentContent = await fetchDocumentPreview(document.id);
        
        // Split the content by newlines and filter out empty lines
        const contentLines = documentContent
          .split('\n')
          .filter(line => line.trim().length > 0);
        
        // If we have content, use it
        if (contentLines.length > 0) {
          setContent(contentLines);
          
          // Store the content in the global store for future use
          addDocumentPreview(document.id, contentLines);
        } else {
          // If no content, show basic document info
          const pageCount = document.page_count || 'unknown number of';
          const basicContent = [
            `Document Name: ${document.name}`,
            `Pages: ${pageCount}`,
            `Author: ${document.author || 'Unknown'}`,
            `Tag: ${document.tag || 'None'}`,
            '',
            'No content could be extracted from this document.'
          ];
          setContent(basicContent);
          
          // Store the basic content in the global store
          addDocumentPreview(document.id, basicContent);
        }
      } catch (err) {
        // Show more detailed error information
        if (err instanceof Error) {
          setError(`Failed to load document preview: ${err.message}`);
        } else {
          setError('Failed to load document preview. Please try again later.');
        }
        // Reset the fetched flag on error so we can try again
        contentFetchedRef.current = false;
      } finally {
        setLoading(false);
      }
    };
    
    fetchContent();
    
  }, [document?.id]); // Only depend on document ID
  
  if (!row || !document) {
    return null;
  }
  
  const pageCount = document.page_count || 'Unknown';
  const author = document.author || 'Unknown';
  const tag = document.tag || '';
  
  return (
    <Drawer
      opened={!!document}
      onClose={onClose}
      title={
        <Group gap="xs">
          <IconFileText size={20} />
          <Text>Document Preview</Text>
        </Group>
      }
      position="right"
      size="xl"
    >
      <Stack>
        <Card withBorder>
          <Title order={4}>{document.name}</Title>
          <Text size="sm" c="dimmed">Pages: {pageCount}</Text>
          <Text size="sm" c="dimmed">Author: {author}</Text>
          {tag && (
            <Text size="sm" c="dimmed">Tag: {tag}</Text>
          )}
        </Card>
        
        {/* Document content - Text view only */}
        <Box>
          <ScrollArea h="calc(100vh - 250px)" type="auto">
            {loading ? (
              <Box ta="center" py="xl">
                <Loader />
                <Text mt="md">Loading document content...</Text>
              </Box>
            ) : error ? (
              <Box ta="center" py="xl">
                <IconX size={40} color="red" />
                <Text mt="md" c="red">{error}</Text>
              </Box>
            ) : content.length > 0 ? (
              <Stack>
                {content.map((text, index) => (
                  <Card key={index} withBorder p="md">
                    <Text>{text}</Text>
                  </Card>
                ))}
              </Stack>
            ) : (
              <Box ta="center" py="xl">
                <Text>No text content available for this document.</Text>
              </Box>
            )}
          </ScrollArea>
        </Box>
        
        <Button fullWidth onClick={onClose}>Close Preview</Button>
      </Stack>
    </Drawer>
  );
}

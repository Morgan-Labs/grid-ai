import { useState } from 'react';
import { Button, Card, Group, Radio, Stack, Text, Title } from '@mantine/core';
import { notifications } from '../utils/notifications';

/**
 * Component for testing error handling and notifications
 */
export function ErrorTest() {
  const [errorType, setErrorType] = useState('timeout');
  const [loading, setLoading] = useState(false);

  const triggerBackendError = async () => {
    setLoading(true);
    try {
      console.log(`Triggering backend error: ${errorType}`);
      const response = await fetch(`http://localhost:8000/api/v1/query/test-error?error_type=${errorType}`);
      
      console.log('Response status:', response.status);
      
      if (!response.ok) {
        let errorMessage = 'Unknown error';
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || 'Unknown error';
          console.log('Error data:', errorData);
        } catch (parseError) {
          console.error('Error parsing error response:', parseError);
          // Try to get text if JSON parsing fails
          const errorText = await response.text();
          errorMessage = errorText || errorMessage;
          console.log('Error text:', errorText);
        }
        
        throw new Error(errorMessage);
      }
      
      const data = await response.json();
      console.log('Success data:', data);
      notifications.show({
        title: 'Success',
        message: data.message,
        color: 'green'
      });
    } catch (error) {
      console.error('Error in triggerBackendError:', error);
      notifications.show({
        title: 'Error',
        message: error instanceof Error ? error.message : 'Unknown error',
        color: 'red'
      });
    } finally {
      setLoading(false);
    }
  };

  const triggerFrontendError = (type: string) => {
    if (type === 'success') {
      notifications.show({
        title: 'Success',
        message: 'This is a success notification',
        color: 'green'
      });
    } else if (type === 'error') {
      notifications.show({
        title: 'Error',
        message: 'This is an error notification',
        color: 'red'
      });
    } else if (type === 'warning') {
      notifications.show({
        title: 'Warning',
        message: 'This is a warning notification',
        color: 'yellow'
      });
    } else if (type === 'info') {
      notifications.show({
        title: 'Info',
        message: 'This is an info notification',
        color: 'blue'
      });
    }
  };

  return (
    <Card shadow="sm" padding="lg" radius="md" withBorder>
      <Title order={3} mb="md">Error Handling Test</Title>
      
      <Stack>
        <Card withBorder>
          <Title order={4} mb="md">Test Backend Errors</Title>
          <Text mb="md">
            Test how the application handles different types of backend errors.
            This will make a request to the backend that will intentionally fail with the selected error type.
          </Text>
          
          <Radio.Group
            value={errorType}
            onChange={setErrorType}
            name="errorType"
            label="Select error type"
            mb="md"
          >
            <Group mt="xs">
              <Radio value="timeout" label="Timeout Error" />
              <Radio value="validation" label="Validation Error" />
              <Radio value="server" label="Server Error" />
            </Group>
          </Radio.Group>
          
          <Button 
            onClick={triggerBackendError} 
            loading={loading}
            color="red"
          >
            Trigger Backend Error
          </Button>
        </Card>
        
        <Card withBorder>
          <Title order={4} mb="md">Test Frontend Notifications</Title>
          <Text mb="md">
            Test different types of notifications without making backend requests.
          </Text>
          
          <Group>
            <Button onClick={() => triggerFrontendError('success')} color="green">
              Success
            </Button>
            <Button onClick={() => triggerFrontendError('error')} color="red">
              Error
            </Button>
            <Button onClick={() => triggerFrontendError('warning')} color="yellow">
              Warning
            </Button>
            <Button onClick={() => triggerFrontendError('info')} color="blue">
              Info
            </Button>
          </Group>
        </Card>
      </Stack>
    </Card>
  );
}

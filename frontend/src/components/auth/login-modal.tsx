import { useState } from 'react';
import { 
  Modal, 
  PasswordInput, 
  Button, 
  Group, 
  Text, 
  Box,
  LoadingOverlay
} from '@mantine/core';
import { useStore } from '@config/store';

interface LoginModalProps {
  opened: boolean;
  onClose: () => void;
}

export function LoginModal({ opened, onClose }: LoginModalProps) {
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  
  const login = useStore(state => state.login);
  const isAuthenticating = useStore(state => state.auth.isAuthenticating);
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!password.trim()) {
      setError('Password is required');
      return;
    }
    
    setError('');
    
    try {
      await login(password);
      onClose();
    } catch (err) {
      // Error notifications are handled in the store
      setPassword('');
    }
  };
  
  return (
    <Modal
      opened={opened}
      onClose={onClose}
      title="Authentication Required"
      centered
      closeOnClickOutside={false}
      closeOnEscape={false}
      withCloseButton={false}
    >
      <Box pos="relative">
        <LoadingOverlay visible={isAuthenticating} />
        
        <form onSubmit={handleSubmit}>
          <Text size="sm" mb="md">
            Please enter the password to access this application.
          </Text>
          
          <PasswordInput
            label="Password"
            placeholder="Enter password"
            value={password}
            onChange={(e) => setPassword(e.currentTarget.value)}
            error={error}
            required
            mb="md"
            autoFocus
          />
          
          <Group justify="flex-end" mt="md">
            <Button type="submit" loading={isAuthenticating}>
              Login
            </Button>
          </Group>
        </form>
      </Box>
    </Modal>
  );
}

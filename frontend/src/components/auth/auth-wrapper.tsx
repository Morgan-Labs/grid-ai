import { useEffect, useState } from 'react';
import { Center, Loader, Alert, Text, Button } from '@mantine/core';
import { IconAlertCircle } from '@tabler/icons-react';
import { useStore } from '@config/store';
import { LoginModal } from './login-modal';

interface AuthWrapperProps {
  children: React.ReactNode;
}

export function AuthWrapper({ children }: AuthWrapperProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [showLogin, setShowLogin] = useState(false);
  const [servicesInitialized, setServicesInitialized] = useState(true);
  
  const isAuthenticated = useStore(state => state.auth.isAuthenticated);
  const token = useStore(state => state.auth.token);
  const checkAuth = useStore(state => state.checkAuth);
  
  useEffect(() => {
    const verifyAuth = async () => {
      setIsLoading(true);
      
      // If we have a token, verify it
      if (token) {
        try {
          const result = await checkAuth();
          
          // Check if services are initialized from the response
          if (result && typeof result === 'object' && 'servicesInitialized' in result) {
            setServicesInitialized(result.servicesInitialized);
          }
          
          if (!result) {
            setShowLogin(true);
          }
        } catch (error) {
          console.error('Error verifying token:', error);
          setShowLogin(true);
        }
      } else {
        // No token, show login
        setShowLogin(true);
      }
      
      setIsLoading(false);
    };
    
    verifyAuth();
  }, [token, checkAuth]);
  
  // Show loading state while checking authentication
  if (isLoading) {
    return (
      <Center style={{ height: '100vh' }}>
        <Loader size="lg" />
      </Center>
    );
  }
  
  // Show login modal if not authenticated
  if (!isAuthenticated) {
    return (
      <LoginModal 
        opened={showLogin} 
        onClose={() => {}} // Empty function since we don't want to close the modal until authenticated
      />
    );
  }
  
  // Check if services are initialized
  if (isAuthenticated && !servicesInitialized) {
    return (
      <div style={{ padding: '20px' }}>
        <Alert 
          icon={<IconAlertCircle size="1.1rem" />} 
          title="Backend Services Not Initialized" 
          color="yellow"
        >
          <Text mb="md">
            The application is authenticated, but the backend services could not be initialized.
            This is likely due to missing API keys or configuration in the backend.
          </Text>
          <Text mb="md">
            You can still use the application, but some features may not work properly.
          </Text>
          <Button color="yellow" onClick={() => window.location.reload()}>
            Retry Connection
          </Button>
        </Alert>
      </div>
    );
  }
  
  // User is authenticated, render children
  return <>{children}</>;
}

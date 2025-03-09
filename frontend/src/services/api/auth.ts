/**
 * Authentication service for the frontend
 */
import { API_ENDPOINTS, DEFAULT_HEADERS } from '../../config/api';

// Authentication error class
export class AuthError extends Error {
  status: number;
  
  constructor(message: string, status: number) {
    super(message);
    this.name = 'AuthError';
    this.status = status;
  }
}

// Token response interface
export interface TokenResponse {
  access_token: string;
  token_type: string;
}

/**
 * Login with password
 * @param password The password to authenticate with
 * @returns The JWT token response
 * @throws AuthError if authentication fails
 */
export const login = async (password: string): Promise<TokenResponse> => {
  try {
    const response = await fetch(`${API_ENDPOINTS.BASE_URL}/api/v1/auth/login`, {
      method: 'POST',
      headers: DEFAULT_HEADERS,
      body: JSON.stringify({ password }),
      credentials: 'include',
    });
    
    if (!response.ok) {
      throw new AuthError(
        response.status === 401 ? 'Incorrect password' : `Authentication failed: ${response.statusText}`,
        response.status
      );
    }
    
    return response.json();
  } catch (error) {
    console.error('Login error:', error);
    if (error instanceof AuthError) {
      throw error;
    }
    throw new AuthError('Authentication failed', 500);
  }
};

/**
 * Verify if the current token is valid
 * @param token The JWT token to verify
 * @returns Response object with authentication status and services initialization status
 */
export const verifyToken = async (token: string): Promise<{ isValid: boolean; servicesInitialized: boolean }> => {
  try {
    const response = await fetch(`${API_ENDPOINTS.BASE_URL}/api/v1/auth/verify`, {
      method: 'POST',
      headers: {
        ...DEFAULT_HEADERS,
        'Authorization': `Bearer ${token}`
      },
      credentials: 'include',
    });
    
    if (!response.ok) {
      return { isValid: false, servicesInitialized: false };
    }
    
    const data = await response.json();
    return { 
      isValid: true, 
      servicesInitialized: data.services_initialized || false 
    };
  } catch (error) {
    console.error('Token verification error:', error);
    return { isValid: false, servicesInitialized: false };
  }
};

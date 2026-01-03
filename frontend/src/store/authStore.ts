'use client';

import { create } from 'zustand';

export interface AuthUser {
  user_id: string;
  email: string;
  name?: string | null;
  created_at: string;
}

export type AuthStatus = 'idle' | 'loading' | 'authenticated';

interface AuthStore {
  token: string | null;
  user: AuthUser | null;
  status: AuthStatus;
  setSession: (token: string, user: AuthUser) => void;
  setUser: (user: AuthUser | null) => void;
  setStatus: (status: AuthStatus) => void;
  clearSession: () => void;
}

const getStoredToken = () => {
  if (typeof window === 'undefined') {
    return null;
  }
  return window.localStorage.getItem('myaos_token');
};

const setStoredToken = (token: string | null) => {
  if (typeof window === 'undefined') {
    return;
  }
  if (!token) {
    window.localStorage.removeItem('myaos_token');
  } else {
    window.localStorage.setItem('myaos_token', token);
  }
};

export const useAuthStore = create<AuthStore>((set) => ({
  token: getStoredToken(),
  user: null,
  status: 'idle',
  setSession: (token, user) => {
    setStoredToken(token);
    set({ token, user, status: 'authenticated' });
  },
  setUser: (user) => {
    set({ user, status: user ? 'authenticated' : 'idle' });
  },
  setStatus: (status) => set({ status }),
  clearSession: () => {
    setStoredToken(null);
    set({ token: null, user: null, status: 'idle' });
  }
}));

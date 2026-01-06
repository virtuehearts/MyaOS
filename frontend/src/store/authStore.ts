'use client';

import { create } from 'zustand';

export interface AuthUser {
  user_id: string;
  email: string;
  name?: string | null;
  created_at: string;
}

export type AuthStatus = 'idle' | 'loading' | 'authenticated';

const USER_ID_STORAGE_KEY = 'myaos_user_id';

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

export const getStoredUserId = () => {
  if (typeof window === 'undefined') {
    return null;
  }
  return window.localStorage.getItem(USER_ID_STORAGE_KEY);
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

const setStoredUserId = (userId: string | null) => {
  if (typeof window === 'undefined') {
    return;
  }
  if (!userId) {
    window.localStorage.removeItem(USER_ID_STORAGE_KEY);
  } else {
    window.localStorage.setItem(USER_ID_STORAGE_KEY, userId);
  }
};

export const createLocalUserId = (input: string) => {
  const normalized = input.trim().toLowerCase();
  if (!normalized) {
    return 'local-guest';
  }
  let hash = 0;
  for (let index = 0; index < normalized.length; index += 1) {
    hash = (hash * 31 + normalized.charCodeAt(index)) >>> 0;
  }
  return `local-${hash.toString(16)}`;
};

export const useAuthStore = create<AuthStore>((set) => ({
  token: getStoredToken(),
  user: null,
  status: 'idle',
  setSession: (token, user) => {
    setStoredToken(token);
    setStoredUserId(user.user_id);
    set({ token, user, status: 'authenticated' });
  },
  setUser: (user) => {
    setStoredUserId(user?.user_id ?? null);
    set({ user, status: user ? 'authenticated' : 'idle' });
  },
  setStatus: (status) => set({ status }),
  clearSession: () => {
    setStoredToken(null);
    setStoredUserId(null);
    set({ token: null, user: null, status: 'idle' });
  }
}));

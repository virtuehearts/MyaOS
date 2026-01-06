'use client';

import { createJSONStorage, type StateStorage } from 'zustand/middleware';

import { getStoredUserId } from '@/store/authStore';

const resolveUserKey = (name: string) => {
  const userId = getStoredUserId() ?? 'anonymous';
  return `${name}:${userId}`;
};

export const createUserScopedStorage = () => {
  if (typeof window === 'undefined') {
    return undefined;
  }
  const storage: StateStorage = {
    getItem: (name) => window.localStorage.getItem(resolveUserKey(name)),
    setItem: (name, value) => window.localStorage.setItem(resolveUserKey(name), value),
    removeItem: (name) => window.localStorage.removeItem(resolveUserKey(name))
  };
  return createJSONStorage(() => storage);
};

export const userScopedStorage = createUserScopedStorage();

'use client';

import { create } from 'zustand';

interface ApiKeyStore {
  apiKey: string;
  setApiKey: (key: string) => void;
  clearApiKey: () => void;
}

const getStoredKey = () => {
  if (typeof window === 'undefined') {
    return '';
  }
  return window.localStorage.getItem('openrouter_api_key') ?? '';
};

const setStoredKey = (key: string) => {
  if (typeof window === 'undefined') {
    return;
  }
  if (!key) {
    window.localStorage.removeItem('openrouter_api_key');
  } else {
    window.localStorage.setItem('openrouter_api_key', key);
  }
};

export const useApiKeyStore = create<ApiKeyStore>((set) => ({
  apiKey: getStoredKey(),
  setApiKey: (key) => {
    setStoredKey(key);
    set({ apiKey: key });
  },
  clearApiKey: () => {
    setStoredKey('');
    set({ apiKey: '' });
  }
}));

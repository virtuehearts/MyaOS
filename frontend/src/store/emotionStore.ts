'use client';

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

import { userScopedStorage } from '@/store/userScopedStorage';
interface EmotionStore {
  emotion: string;
  setEmotion: (emotion: string) => void;
  resetEmotion: () => void;
}

export const useEmotionStore = create<EmotionStore>()(
  persist(
    (set) => ({
      emotion: 'neutral',
      setEmotion: (emotion) => set({ emotion }),
      resetEmotion: () => set({ emotion: 'neutral' })
    }),
    {
      name: 'mya-emotion-store',
      storage: userScopedStorage,
      partialize: (state) => ({ emotion: state.emotion })
    }
  )
);

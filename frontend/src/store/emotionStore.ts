'use client';

import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

interface EmotionStore {
  emotion: string;
  setEmotion: (emotion: string) => void;
}

const storage =
  typeof window === 'undefined'
    ? undefined
    : createJSONStorage(() => window.localStorage);

export const useEmotionStore = create<EmotionStore>()(
  persist(
    (set) => ({
      emotion: 'neutral',
      setEmotion: (emotion) => set({ emotion })
    }),
    {
      name: 'mya-emotion-store',
      storage,
      partialize: (state) => ({ emotion: state.emotion })
    }
  )
);

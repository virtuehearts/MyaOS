import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

interface UiStore {
  showSystemInsights: boolean;
  setShowSystemInsights: (value: boolean) => void;
}

const storage =
  typeof window === 'undefined'
    ? undefined
    : createJSONStorage(() => window.localStorage);

export const useUiStore = create<UiStore>()(
  persist(
    (set) => ({
      showSystemInsights: false,
      setShowSystemInsights: (value) => set({ showSystemInsights: value })
    }),
    {
      name: 'mya-ui-store',
      storage,
      partialize: (state) => ({
        showSystemInsights: state.showSystemInsights
      })
    }
  )
);

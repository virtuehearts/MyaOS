import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

export type WindowId = 'chat' | 'memory' | 'settings' | 'console' | 'my-chats';

export interface OsWindowState {
  id: WindowId;
  title: string;
  isOpen: boolean;
  isMinimized: boolean;
  isResizing: boolean;
  zIndex: number;
  width: number;
  height: number;
  minWidth: number;
  minHeight: number;
}

interface OsStore {
  windows: OsWindowState[];
  activeWindowId: WindowId | null;
  openWindow: (id: WindowId) => void;
  closeWindow: (id: WindowId) => void;
  focusWindow: (id: WindowId) => void;
  toggleMinimize: (id: WindowId) => void;
  setWindowSize: (id: WindowId, size: { width: number; height: number }) => void;
  setWindowResizing: (id: WindowId, isResizing: boolean) => void;
}

const baseWindows: OsWindowState[] = [
  {
    id: 'chat',
    title: 'Mya Chat',
    isOpen: true,
    isMinimized: false,
    isResizing: false,
    zIndex: 10,
    width: 560,
    height: 420,
    minWidth: 360,
    minHeight: 280
  },
  {
    id: 'memory',
    title: 'Memory Vault',
    isOpen: false,
    isMinimized: false,
    isResizing: false,
    zIndex: 8,
    width: 520,
    height: 380,
    minWidth: 360,
    minHeight: 280
  },
  {
    id: 'settings',
    title: 'System Settings',
    isOpen: false,
    isMinimized: false,
    isResizing: false,
    zIndex: 7,
    width: 540,
    height: 400,
    minWidth: 360,
    minHeight: 280
  },
  {
    id: 'console',
    title: 'Console',
    isOpen: false,
    isMinimized: false,
    isResizing: false,
    zIndex: 6,
    width: 520,
    height: 320,
    minWidth: 360,
    minHeight: 220
  },
  {
    id: 'my-chats',
    title: 'My Chats',
    isOpen: false,
    isMinimized: false,
    isResizing: false,
    zIndex: 5,
    width: 520,
    height: 380,
    minWidth: 360,
    minHeight: 260
  }
];

const storage =
  typeof window === 'undefined'
    ? undefined
    : createJSONStorage(() => window.localStorage);

export const useOsStore = create<OsStore>()(
  persist(
    (set) => ({
      windows: baseWindows,
      activeWindowId: 'chat',
      openWindow: (id) =>
        set((state) => ({
          windows: state.windows.map((window) =>
            window.id === id
              ? { ...window, isOpen: true, isMinimized: false, zIndex: 20 }
              : window
          ),
          activeWindowId: id
        })),
      closeWindow: (id) =>
        set((state) => ({
          windows: state.windows.map((window) =>
            window.id === id ? { ...window, isOpen: false } : window
          ),
          activeWindowId: state.activeWindowId === id ? null : state.activeWindowId
        })),
      focusWindow: (id) =>
        set((state) => ({
          windows: state.windows.map((window) =>
            window.id === id
              ? { ...window, zIndex: 30, isMinimized: false }
              : window
          ),
          activeWindowId: id
        })),
      toggleMinimize: (id) =>
        set((state) => ({
          windows: state.windows.map((window) =>
            window.id === id
              ? { ...window, isMinimized: !window.isMinimized }
              : window
          )
        })),
      setWindowSize: (id, size) =>
        set((state) => ({
          windows: state.windows.map((window) =>
            window.id === id ? { ...window, ...size } : window
          )
        })),
      setWindowResizing: (id, isResizing) =>
        set((state) => ({
          windows: state.windows.map((window) =>
            window.id === id ? { ...window, isResizing } : window
          )
        }))
    }),
    {
      name: 'mya-os-store',
      storage,
      partialize: (state) => ({
        windows: state.windows,
        activeWindowId: state.activeWindowId
      })
    }
  )
);

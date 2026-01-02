import { create } from 'zustand';

export type WindowId = 'chat' | 'memory' | 'settings';

export interface OsWindowState {
  id: WindowId;
  title: string;
  isOpen: boolean;
  isMinimized: boolean;
  zIndex: number;
}

interface OsStore {
  windows: OsWindowState[];
  activeWindowId: WindowId | null;
  openWindow: (id: WindowId) => void;
  closeWindow: (id: WindowId) => void;
  focusWindow: (id: WindowId) => void;
  toggleMinimize: (id: WindowId) => void;
}

const baseWindows: OsWindowState[] = [
  { id: 'chat', title: 'Mya Chat', isOpen: true, isMinimized: false, zIndex: 10 },
  { id: 'memory', title: 'Memory Vault', isOpen: false, isMinimized: false, zIndex: 8 },
  { id: 'settings', title: 'System Settings', isOpen: false, isMinimized: false, zIndex: 7 }
];

export const useOsStore = create<OsStore>((set) => ({
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
        window.id === id ? { ...window, zIndex: 30, isMinimized: false } : window
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
    }))
}));

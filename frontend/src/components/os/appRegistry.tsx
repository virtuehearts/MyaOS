import type { ReactNode } from 'react';

import { ChatWindow } from '@/components/os/ChatWindow';
import { ConsoleWindow } from '@/components/os/ConsoleWindow';
import { MemoryWindow } from '@/components/os/MemoryWindow';
import { SettingsWindow } from '@/components/os/SettingsWindow';
import type { WindowId } from '@/store/osStore';

export interface AppRenderContext {
  desktopBackground: string | null;
  onBackgroundSelect: (file: File) => void;
  onClearBackground: () => void;
}

export interface AppRegistryEntry {
  id: WindowId;
  label: string;
  title: string;
  description: string;
  route: string;
  accent: string;
  startMenuSection?: 'primary' | 'secondary';
  pinned?: boolean;
  desktop?: boolean;
  render: (context: AppRenderContext) => ReactNode;
}

export const appRegistry: AppRegistryEntry[] = [
  {
    id: 'chat',
    label: 'Mya Chat',
    title: 'Mya Chat',
    description: 'Companion messaging',
    route: '/apps/chat',
    accent: 'bg-retro-title-active',
    startMenuSection: 'primary',
    pinned: true,
    desktop: true,
    render: () => <ChatWindow />
  },
  {
    id: 'memory',
    label: 'Memory Vault',
    title: 'Memory Vault',
    description: 'Recall snippets',
    route: '/apps/memory',
    accent: 'bg-retro-accent',
    startMenuSection: 'primary',
    pinned: true,
    desktop: true,
    render: () => <MemoryWindow />
  },
  {
    id: 'settings',
    label: 'System Settings',
    title: 'System Settings',
    description: 'Tweak the OS',
    route: '/apps/settings',
    accent: 'bg-retro-border',
    startMenuSection: 'primary',
    pinned: true,
    desktop: true,
    render: ({ desktopBackground, onBackgroundSelect, onClearBackground }) => (
      <SettingsWindow
        desktopBackground={desktopBackground}
        onBackgroundSelect={onBackgroundSelect}
        onClearBackground={onClearBackground}
      />
    )
  },
  {
    id: 'console',
    label: 'Console',
    title: 'Console',
    description: 'System diagnostics',
    route: '/apps/console',
    accent: 'bg-retro-title-active',
    startMenuSection: 'primary',
    pinned: false,
    desktop: false,
    render: () => <ConsoleWindow />
  }
];

export const appRegistryById = new Map(appRegistry.map((app) => [app.id, app]));

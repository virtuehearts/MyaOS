import type { ReactNode } from 'react';
import {
  Archive,
  Folder,
  MessageCircle,
  MessagesSquare,
  Settings,
  Terminal
} from 'lucide-react';

import { ChatWindow } from '@/components/os/ChatWindow';
import { ConsoleWindow } from '@/components/os/ConsoleWindow';
import { FileManagerWindow } from '@/components/os/FileManagerWindow';
import { MyChatsWindow } from '@/components/os/MyChatsWindow';
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
  icon: ReactNode;
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
    icon: <MessageCircle className="h-4 w-4" />,
    startMenuSection: 'primary',
    pinned: false,
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
    icon: <Archive className="h-4 w-4" />,
    startMenuSection: 'primary',
    pinned: false,
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
    icon: <Settings className="h-4 w-4" />,
    startMenuSection: 'primary',
    pinned: false,
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
    icon: <Terminal className="h-4 w-4" />,
    startMenuSection: 'primary',
    pinned: false,
    desktop: true,
    render: () => <ConsoleWindow />
  },
  {
    id: 'my-chats',
    label: 'My Chats',
    title: 'My Chats',
    description: 'Manage sessions',
    route: '/apps/my-chats',
    accent: 'bg-retro-accent',
    icon: <MessagesSquare className="h-4 w-4" />,
    startMenuSection: 'primary',
    pinned: false,
    desktop: true,
    render: () => <MyChatsWindow />
  },
  {
    id: 'file-manager',
    label: 'File Manager',
    title: 'File Manager',
    description: 'Browse system assets',
    route: '/apps/file-manager',
    accent: 'bg-retro-border',
    icon: <Folder className="h-4 w-4" />,
    startMenuSection: 'primary',
    pinned: false,
    desktop: true,
    render: () => <FileManagerWindow />
  }
];

export const appRegistryById = new Map(appRegistry.map((app) => [app.id, app]));

'use client';

import { Button } from '@/components/ui/button';
import { apiRequest } from '@/lib/api';
import { useAuthStore } from '@/store/authStore';
import { useOsStore } from '@/store/osStore';

interface StartMenuProps {
  isOpen: boolean;
  onClose: () => void;
}

export function StartMenu({ isOpen, onClose }: StartMenuProps) {
  const { user, token, clearSession } = useAuthStore();
  const { openWindow, focusWindow } = useOsStore();

  if (!isOpen) {
    return null;
  }

  const handleSignOut = async () => {
    try {
      if (token) {
        await apiRequest('/auth/logout', {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}` }
        });
      }
    } finally {
      clearSession();
      onClose();
    }
  };

  const handleLaunch = (id: 'chat' | 'memory' | 'settings') => {
    openWindow(id);
    focusWindow(id);
    onClose();
  };

  return (
    <div className="fixed bottom-12 left-3 z-40 w-[240px]">
      <div className="mya-panel border border-retro-border">
        <div className="border-b border-retro-border bg-retro-title-active px-3 py-2 text-xs font-semibold text-retro-text">
          {user?.name ? `${user.name}` : 'MyaOS User'}
        </div>
        <div className="flex flex-col py-2 text-xs">
          <div className="px-3 pb-1 text-[10px] uppercase text-retro-accent">
            Local Programs
          </div>
          <Button variant="ghost" className="justify-start px-3" onClick={() => handleLaunch('chat')}>
            Mya Chat
          </Button>
          <Button variant="ghost" className="justify-start px-3" onClick={() => handleLaunch('memory')}>
            Memory Vault
          </Button>
          <Button
            variant="ghost"
            className="justify-start px-3"
            onClick={() => handleLaunch('settings')}
          >
            System Settings
          </Button>
          <div className="mt-2 px-3 pb-1 text-[10px] uppercase text-retro-accent">
            Remote Programs (Python)
          </div>
          <Button variant="ghost" className="justify-start px-3" onClick={onClose}>
            Diagnostics Runner
          </Button>
          <Button variant="ghost" className="justify-start px-3" onClick={onClose}>
            Report Sync Service
          </Button>
          <Button variant="ghost" className="justify-start px-3" onClick={onClose}>
            Data Import Console
          </Button>
          <div className="my-2 border-t border-retro-border" />
          <Button variant="ghost" className="justify-start px-3" onClick={handleSignOut}>
            Shut Down...
          </Button>
        </div>
      </div>
    </div>
  );
}

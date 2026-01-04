'use client';

import { Button } from '@/components/ui/button';
import { apiRequest } from '@/lib/api';
import { cn } from '@/lib/utils';
import { useAuthStore } from '@/store/authStore';
import { useOsStore, type WindowId } from '@/store/osStore';
import { appRegistry } from '@/components/os/appRegistry';
import { useEffect, useMemo, useState, type CSSProperties, type RefObject } from 'react';

interface StartMenuProps {
  isOpen: boolean;
  onClose: () => void;
  anchorRef: RefObject<HTMLButtonElement>;
}

const comingSoonApps = [
  {
    id: 'calculator',
    label: 'Calculator',
    description: 'Coming soon',
    windowId: null as WindowId | null,
    accent: 'bg-retro-title-active'
  },
  {
    id: 'notes',
    label: 'Notes',
    description: 'Coming soon',
    windowId: null as WindowId | null,
    accent: 'bg-retro-accent'
  }
] as const;

type StartMenuApp = {
  id: string;
  label: string;
  description: string;
  accent: string;
  windowId: WindowId | null;
  disabled: boolean;
};

export function StartMenu({ isOpen, onClose, anchorRef }: StartMenuProps) {
  const { user, token, clearSession } = useAuthStore();
  const { openWindow, focusWindow } = useOsStore();
  const [anchorStyle, setAnchorStyle] = useState<CSSProperties | null>(null);

  const menuApps = useMemo(() => {
    const registryApps: StartMenuApp[] = appRegistry
      .filter((app) => app.startMenuSection === 'primary')
      .map((app) => ({
        id: app.id,
        label: app.label,
        description: app.description,
        accent: app.accent,
        windowId: app.id,
        disabled: false
      }));
    const comingSoon: StartMenuApp[] = comingSoonApps.map((app) => ({
      id: app.id,
      label: app.label,
      description: app.description,
      accent: app.accent,
      windowId: app.windowId,
      disabled: true
    }));
    return { registryApps, comingSoon };
  }, []);

  useEffect(() => {
    const updateAnchor = () => {
      const anchor = anchorRef.current;
      if (!anchor) {
        return;
      }
      const rect = anchor.getBoundingClientRect();
      setAnchorStyle({
        left: rect.left,
        bottom: window.innerHeight - rect.top + 8
      });
    };

    updateAnchor();
    if (!isOpen) {
      return;
    }

    window.addEventListener('resize', updateAnchor);
    window.addEventListener('scroll', updateAnchor);
    return () => {
      window.removeEventListener('resize', updateAnchor);
      window.removeEventListener('scroll', updateAnchor);
    };
  }, [anchorRef, isOpen]);

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

  const handleLaunch = (app: StartMenuApp) => {
    if (!app.windowId) {
      return;
    }
    openWindow(app.windowId);
    focusWindow(app.windowId);
    onClose();
  };

  return (
    <div
      className={cn(
        'fixed z-40 w-[260px] origin-bottom-left transition-all duration-200',
        isOpen
          ? 'pointer-events-auto scale-100 opacity-100 translate-y-0'
          : 'pointer-events-none scale-95 opacity-0 translate-y-2'
      )}
      style={anchorStyle ?? { left: 12, bottom: 52 }}
    >
      <div className="mya-panel border border-retro-border shadow-[0_8px_0_rgba(0,0,0,0.15)]">
        <div className="border-b border-retro-border bg-retro-title-active px-3 py-2 text-xs font-semibold text-retro-text">
          {user?.name ? `${user.name}` : 'MyaOS User'}
        </div>
        <div className="flex flex-col py-2 text-xs">
          <div className="px-3 pb-1 text-[10px] uppercase text-retro-accent">
            Local Programs
          </div>
          {menuApps.registryApps.map((app) => (
            <Button
              key={app.id}
              variant="ghost"
              className="justify-start px-3"
              onClick={() => handleLaunch(app)}
              disabled={app.disabled}
            >
              <span className="flex items-center gap-2">
                <span
                  className={cn(
                    'h-5 w-5 border border-retro-border',
                    app.accent
                  )}
                />
                <span className="flex flex-col">
                  <span>{app.label}</span>
                  <span className="text-[10px] text-retro-accent">{app.description}</span>
                </span>
              </span>
            </Button>
          ))}
          <div className="mt-2 px-3 pb-1 text-[10px] uppercase text-retro-accent">
            Coming Soon
          </div>
          {menuApps.comingSoon.map((app) => (
            <Button
              key={app.id}
              variant="ghost"
              className="justify-start px-3 opacity-70"
              onClick={() => handleLaunch(app)}
              disabled={app.disabled}
            >
              <span className="flex items-center gap-2">
                <span
                  className={cn(
                    'h-5 w-5 border border-retro-border',
                    app.accent
                  )}
                />
                <span className="flex flex-col">
                  <span>{app.label}</span>
                  <span className="text-[10px] text-retro-accent">{app.description}</span>
                </span>
              </span>
            </Button>
          ))}
          <div className="my-2 border-t border-retro-border" />
          <Button variant="ghost" className="justify-start px-3" onClick={handleSignOut}>
            Shut Down...
          </Button>
        </div>
      </div>
    </div>
  );
}

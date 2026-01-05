'use client';

import { Button } from '@/components/ui/button';
import { appRegistry } from '@/components/os/appRegistry';
import { cn } from '@/lib/utils';
import { useOsStore, type WindowId } from '@/store/osStore';
import type { Ref } from 'react';

interface TaskbarProps {
  isStartOpen: boolean;
  onToggleStart: () => void;
  isStartDisabled?: boolean;
  startButtonRef?: Ref<HTMLButtonElement>;
}

export function Taskbar({
  isStartOpen,
  onToggleStart,
  isStartDisabled,
  startButtonRef
}: TaskbarProps) {
  const { windows, openWindow, focusWindow } = useOsStore();

  const handleLaunch = (id: WindowId) => {
    const target = windows.find((window) => window.id === id);
    if (!target) {
      return;
    }
    if (!target.isOpen) {
      openWindow(id);
      return;
    }
    focusWindow(id);
  };

  const now = new Date();
  const clockLabel = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const pinnedApps = appRegistry.filter((app) => app.pinned);

  return (
    <footer className="fixed bottom-0 left-0 z-50 w-full border-t border-retro-border bg-retro-surface">
      <div className="flex items-center gap-3 px-3 py-2 text-xs text-retro-text">
        <Button
          variant="outline"
          className={cn('h-8 px-3 font-semibold', isStartOpen && 'bg-retro-title-active')}
          onClick={onToggleStart}
          disabled={isStartDisabled}
          ref={startButtonRef}
        >
          MyaOS
        </Button>
        <div className="flex flex-1 items-center justify-center gap-2">
          {pinnedApps.map((app) => (
            <Button
              key={app.id}
              variant="ghost"
              className="h-8 w-8 p-0"
              onClick={() => handleLaunch(app.id)}
              title={app.label}
              aria-label={app.label}
            >
              {app.icon}
              <span className="sr-only">{app.label}</span>
            </Button>
          ))}
        </div>
        <div className="flex items-center gap-3 text-xs">
          <span className="border border-retro-border bg-retro-bg px-2 py-1">{clockLabel}</span>
        </div>
      </div>
    </footer>
  );
}

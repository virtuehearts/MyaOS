'use client';

import { Clock, MessageCircle, Settings, Sparkles } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { useOsStore, type WindowId } from '@/store/osStore';

const pinnedApps: Array<{ id: WindowId; label: string; icon: React.ReactNode }> = [
  { id: 'chat', label: 'Chat', icon: <MessageCircle className="h-4 w-4" /> },
  { id: 'memory', label: 'Memory', icon: <Sparkles className="h-4 w-4" /> },
  { id: 'settings', label: 'Settings', icon: <Settings className="h-4 w-4" /> }
];

export function Taskbar() {
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

  return (
    <footer className="fixed bottom-4 left-1/2 z-50 w-[min(720px,90%)] -translate-x-1/2">
      <div className="mya-panel mya-glow flex items-center justify-between rounded-2xl px-4 py-2">
        <div className="flex items-center gap-2">
          {pinnedApps.map((app) => (
            <Button
              key={app.id}
              variant="ghost"
              className={cn('gap-2 rounded-xl', 'text-slate-200 hover:text-white')}
              onClick={() => handleLaunch(app.id)}
            >
              {app.icon}
              {app.label}
            </Button>
          ))}
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-300">
          <Clock className="h-4 w-4" />
          {clockLabel}
        </div>
      </div>
    </footer>
  );
}

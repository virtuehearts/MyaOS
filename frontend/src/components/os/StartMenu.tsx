'use client';

import { UserCircle2 } from 'lucide-react';

import { Button } from '@/components/ui/button';

interface StartMenuProps {
  isOpen: boolean;
  onClose: () => void;
}

export function StartMenu({ isOpen, onClose }: StartMenuProps) {
  if (!isOpen) {
    return null;
  }

  return (
    <div className="fixed bottom-24 left-1/2 z-40 w-[min(420px,90%)] -translate-x-1/2">
      <div className="mya-panel rounded-2xl p-6 shadow-xl">
        <div className="flex items-center gap-3">
          <UserCircle2 className="h-10 w-10 text-mya-saffron" />
          <div>
            <p className="text-base font-semibold">Welcome back</p>
            <p className="text-xs text-slate-400">Virtueism guided profile</p>
          </div>
        </div>
        <div className="mt-6 space-y-2 text-sm">
          <Button variant="outline" className="w-full" onClick={onClose}>
            Launch Mya Chat
          </Button>
          <Button variant="outline" className="w-full" onClick={onClose}>
            Open Memory Vault
          </Button>
          <Button variant="ghost" className="w-full" onClick={onClose}>
            System Preferences
          </Button>
        </div>
      </div>
    </div>
  );
}

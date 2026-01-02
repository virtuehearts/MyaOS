'use client';

import { useState } from 'react';

import { ChatWindow } from '@/components/os/ChatWindow';
import { OsWindow } from '@/components/os/OsWindow';
import { StartMenu } from '@/components/os/StartMenu';
import { Taskbar } from '@/components/os/Taskbar';
import { useOsStore } from '@/store/osStore';

export function OsShell() {
  const { windows, closeWindow, focusWindow, toggleMinimize } = useOsStore();
  const [startMenuOpen, setStartMenuOpen] = useState(false);

  const chatWindow = windows.find((window) => window.id === 'chat');

  return (
    <div className="relative min-h-screen overflow-hidden p-8">
      <div className="absolute inset-0 -z-10 opacity-20">
        <div className="h-full w-full bg-[radial-gradient(circle_at_top,_rgba(246,178,62,0.25),_transparent_60%)]" />
      </div>

      {chatWindow?.isOpen && (
        <OsWindow
          id="chat"
          title="Mya Chat"
          isActive={chatWindow.id === 'chat'}
          isMinimized={chatWindow.isMinimized}
          zIndex={chatWindow.zIndex}
          onClose={() => closeWindow('chat')}
          onMinimize={() => toggleMinimize('chat')}
          onFocus={() => focusWindow('chat')}
        >
          <ChatWindow />
        </OsWindow>
      )}

      <button
        type="button"
        className="fixed bottom-24 left-1/2 z-50 -translate-x-1/2 rounded-full bg-mya-saffron px-4 py-2 text-sm font-semibold text-slate-900 shadow-lg"
        onClick={() => setStartMenuOpen((open) => !open)}
      >
        Open Start Menu
      </button>

      <StartMenu isOpen={startMenuOpen} onClose={() => setStartMenuOpen(false)} />
      <Taskbar />
    </div>
  );
}

'use client';

import { useEffect, useState } from 'react';

import { AuthOverlay } from '@/components/os/AuthOverlay';
import { ChatWindow } from '@/components/os/ChatWindow';
import { OsWindow } from '@/components/os/OsWindow';
import { StartMenu } from '@/components/os/StartMenu';
import { Taskbar } from '@/components/os/Taskbar';
import { apiRequest } from '@/lib/api';
import { AuthUser, useAuthStore } from '@/store/authStore';
import { useOsStore } from '@/store/osStore';

export function OsShell() {
  const { windows, closeWindow, focusWindow, toggleMinimize } = useOsStore();
  const [startMenuOpen, setStartMenuOpen] = useState(false);
  const { token, user, setUser, clearSession, status, setStatus } = useAuthStore();

  const chatWindow = windows.find((window) => window.id === 'chat');

  useEffect(() => {
    if (!token || user) {
      return;
    }
    let isMounted = true;
    const fetchUser = async () => {
      setStatus('loading');
      try {
        const response = await apiRequest<{ user: AuthUser }>('/auth/me', {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (isMounted) {
          setUser(response.user);
        }
      } catch {
        if (isMounted) {
          clearSession();
        }
      }
    };
    fetchUser();
    return () => {
      isMounted = false;
    };
  }, [token, user, setUser, clearSession, setStatus]);

  return (
    <div className="relative min-h-screen overflow-hidden p-8">
      <div className="absolute inset-0 -z-10 opacity-20">
        <div className="h-full w-full bg-[radial-gradient(circle_at_top,_rgba(246,178,62,0.25),_transparent_60%)]" />
      </div>

      {!user && status !== 'loading' && <AuthOverlay />}
      {!user && status === 'loading' && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/70 text-sm text-slate-200">
          Restoring your secure session…
        </div>
      )}

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
        disabled={!user}
      >
        Open Start Menu
      </button>

      <StartMenu isOpen={startMenuOpen} onClose={() => setStartMenuOpen(false)} />
      <Taskbar />
    </div>
  );
}

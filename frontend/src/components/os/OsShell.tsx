'use client';

import { useEffect, useRef, useState } from 'react';

import { AuthOverlay } from '@/components/os/AuthOverlay';
import { ChatWindow } from '@/components/os/ChatWindow';
import { MemoryWindow } from '@/components/os/MemoryWindow';
import { OsWindow } from '@/components/os/OsWindow';
import { SettingsWindow } from '@/components/os/SettingsWindow';
import { StartMenu } from '@/components/os/StartMenu';
import { Taskbar } from '@/components/os/Taskbar';
import { apiRequest } from '@/lib/api';
import { AuthUser, useAuthStore } from '@/store/authStore';
import { useOsStore, type WindowId } from '@/store/osStore';

export function OsShell() {
  const {
    windows,
    closeWindow,
    focusWindow,
    toggleMinimize,
    openWindow,
    setWindowResizing,
    setWindowSize
  } = useOsStore();
  const [startMenuOpen, setStartMenuOpen] = useState(false);
  const [desktopBackground, setDesktopBackground] = useState<string | null>(null);
  const { token, user, setUser, clearSession, status, setStatus } = useAuthStore();
  const startButtonRef = useRef<HTMLButtonElement>(null);

  const chatWindow = windows.find((window) => window.id === 'chat');
  const memoryWindow = windows.find((window) => window.id === 'memory');
  const settingsWindow = windows.find((window) => window.id === 'settings');

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

  useEffect(() => {
    return () => {
      if (desktopBackground) {
        URL.revokeObjectURL(desktopBackground);
      }
    };
  }, [desktopBackground]);

  const handleDesktopLaunch = (id: WindowId) => {
    openWindow(id);
    focusWindow(id);
  };

  return (
    <div
      className="relative min-h-[100dvh] w-full overflow-hidden bg-retro-bg p-6"
      style={
        desktopBackground
          ? {
              backgroundImage: `url(${desktopBackground})`,
              backgroundSize: 'cover',
              backgroundPosition: 'center',
              backgroundRepeat: 'no-repeat'
            }
          : undefined
      }
    >

      {!user && status !== 'loading' && <AuthOverlay />}
      {!user && status === 'loading' && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-retro-bg/90 text-sm text-retro-text">
          Restoring your secure session…
        </div>
      )}

      {chatWindow?.isOpen && (
        <OsWindow
          id="chat"
          title="Mya Chat"
          isActive={chatWindow.id === 'chat'}
          isMinimized={chatWindow.isMinimized}
          width={chatWindow.width}
          height={chatWindow.height}
          minWidth={chatWindow.minWidth}
          minHeight={chatWindow.minHeight}
          zIndex={chatWindow.zIndex}
          onClose={() => closeWindow('chat')}
          onMinimize={() => toggleMinimize('chat')}
          onFocus={() => focusWindow('chat')}
          onResizeStart={() => setWindowResizing('chat', true)}
          onResizeStop={(size) => {
            setWindowSize('chat', size);
            setWindowResizing('chat', false);
          }}
        >
          <ChatWindow />
        </OsWindow>
      )}

      {memoryWindow?.isOpen && (
        <OsWindow
          id="memory"
          title="Memory Vault"
          isActive={memoryWindow.id === 'memory'}
          isMinimized={memoryWindow.isMinimized}
          width={memoryWindow.width}
          height={memoryWindow.height}
          minWidth={memoryWindow.minWidth}
          minHeight={memoryWindow.minHeight}
          zIndex={memoryWindow.zIndex}
          onClose={() => closeWindow('memory')}
          onMinimize={() => toggleMinimize('memory')}
          onFocus={() => focusWindow('memory')}
          onResizeStart={() => setWindowResizing('memory', true)}
          onResizeStop={(size) => {
            setWindowSize('memory', size);
            setWindowResizing('memory', false);
          }}
        >
          <MemoryWindow />
        </OsWindow>
      )}

      {settingsWindow?.isOpen && (
        <OsWindow
          id="settings"
          title="System Settings"
          isActive={settingsWindow.id === 'settings'}
          isMinimized={settingsWindow.isMinimized}
          width={settingsWindow.width}
          height={settingsWindow.height}
          minWidth={settingsWindow.minWidth}
          minHeight={settingsWindow.minHeight}
          zIndex={settingsWindow.zIndex}
          onClose={() => closeWindow('settings')}
          onMinimize={() => toggleMinimize('settings')}
          onFocus={() => focusWindow('settings')}
          onResizeStart={() => setWindowResizing('settings', true)}
          onResizeStop={(size) => {
            setWindowSize('settings', size);
            setWindowResizing('settings', false);
          }}
        >
          <SettingsWindow
            desktopBackground={desktopBackground}
            onBackgroundSelect={(file) => {
              const url = URL.createObjectURL(file);
              setDesktopBackground((prev) => {
                if (prev) {
                  URL.revokeObjectURL(prev);
                }
                return url;
              });
            }}
            onClearBackground={() => {
              setDesktopBackground((prev) => {
                if (prev) {
                  URL.revokeObjectURL(prev);
                }
                return null;
              });
            }}
          />
        </OsWindow>
      )}

      <div className="absolute left-4 top-4 flex flex-col gap-4 text-xs text-retro-text">
        <button
          type="button"
          className="flex w-20 flex-col items-center gap-2 border border-retro-border bg-retro-surface px-2 py-2"
          onClick={() => handleDesktopLaunch('chat')}
        >
          <span className="h-8 w-8 border border-retro-border bg-retro-title-active" />
          <span>Chat</span>
        </button>
        <button
          type="button"
          className="flex w-20 flex-col items-center gap-2 border border-retro-border bg-retro-surface px-2 py-2"
          onClick={() => handleDesktopLaunch('memory')}
        >
          <span className="h-8 w-8 border border-retro-border bg-retro-title-active" />
          <span>Memory</span>
        </button>
        <button
          type="button"
          className="flex w-20 flex-col items-center gap-2 border border-retro-border bg-retro-surface px-2 py-2"
          onClick={() => handleDesktopLaunch('settings')}
        >
          <span className="h-8 w-8 border border-retro-border bg-retro-title-active" />
          <span>Settings</span>
        </button>
      </div>

      <StartMenu
        isOpen={startMenuOpen}
        onClose={() => setStartMenuOpen(false)}
        anchorRef={startButtonRef}
      />
      <Taskbar
        isStartOpen={startMenuOpen}
        onToggleStart={() => setStartMenuOpen((open) => !open)}
        isStartDisabled={!user}
        startButtonRef={startButtonRef}
      />
    </div>
  );
}

'use client';

import { useEffect, useRef, useState } from 'react';

import { AuthOverlay } from '@/components/os/AuthOverlay';
import { appRegistry, appRegistryById } from '@/components/os/appRegistry';
import { OsWindow } from '@/components/os/OsWindow';
import { StartMenu } from '@/components/os/StartMenu';
import { Taskbar } from '@/components/os/Taskbar';
import { apiRequest } from '@/lib/api';
import { AuthUser, useAuthStore } from '@/store/authStore';
import { useChatStore } from '@/store/chatStore';
import { useOsStore, type WindowId } from '@/store/osStore';

export function OsShell() {
  const {
    windows,
    activeWindowId,
    closeWindow,
    focusWindow,
    toggleMinimize,
    openWindow,
    setWindowResizing,
    setWindowSize
  } = useOsStore();
  const { markChatClosed } = useChatStore();
  const [startMenuOpen, setStartMenuOpen] = useState(false);
  const [desktopBackground, setDesktopBackground] = useState<string | null>(null);
  const { token, user, setUser, clearSession, status, setStatus } = useAuthStore();
  const startButtonRef = useRef<HTMLButtonElement>(null);

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

  const handleCloseWindow = (id: WindowId) => {
    if (id === 'chat') {
      markChatClosed();
    }
    closeWindow(id);
  };

  const windowContext = {
    desktopBackground,
    onBackgroundSelect: (file: File) => {
      const url = URL.createObjectURL(file);
      setDesktopBackground((prev) => {
        if (prev) {
          URL.revokeObjectURL(prev);
        }
        return url;
      });
    },
    onClearBackground: () => {
      setDesktopBackground((prev) => {
        if (prev) {
          URL.revokeObjectURL(prev);
        }
        return null;
      });
    }
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

      {windows.map((window) => {
        if (!window.isOpen) {
          return null;
        }
        const entry = appRegistryById.get(window.id);
        if (!entry) {
          return null;
        }
        return (
          <OsWindow
            key={window.id}
            id={window.id}
            title={entry.title}
            isActive={activeWindowId === window.id}
            isMinimized={window.isMinimized}
            width={window.width}
            height={window.height}
            minWidth={window.minWidth}
            minHeight={window.minHeight}
            zIndex={window.zIndex}
            onClose={() => handleCloseWindow(window.id)}
            onMinimize={() => toggleMinimize(window.id)}
            onFocus={() => focusWindow(window.id)}
            onResizeStart={() => setWindowResizing(window.id, true)}
            onResizeStop={(size) => {
              setWindowSize(window.id, size);
              setWindowResizing(window.id, false);
            }}
          >
            {entry.render(windowContext)}
          </OsWindow>
        );
      })}

      <div className="absolute left-4 top-4 flex flex-col gap-4 text-xs text-retro-text">
        {appRegistry
          .filter((app) => app.desktop)
          .map((app) => (
            <button
              key={app.id}
              type="button"
              className="flex w-20 flex-col items-center gap-2 border border-retro-border bg-retro-surface px-2 py-2"
              onClick={() => handleDesktopLaunch(app.id)}
            >
              <span className={`h-8 w-8 border border-retro-border ${app.accent}`} />
              <span>{app.label}</span>
            </button>
          ))}
      </div>

      <StartMenu
        isOpen={startMenuOpen}
        onClose={() => setStartMenuOpen(false)}
        anchorRef={startButtonRef}
      />
      <Taskbar
        isStartOpen={startMenuOpen}
        onToggleStart={() => setStartMenuOpen((open) => !open)}
        isStartDisabled={!user && !token}
        startButtonRef={startButtonRef}
      />
    </div>
  );
}

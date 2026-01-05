'use client';

import { useEffect, useMemo, useState } from 'react';
import JSZip from 'jszip';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';
import { useChatStore } from '@/store/chatStore';
import { useOsStore } from '@/store/osStore';

const formatTimestamp = (timestamp: number) =>
  new Date(timestamp).toLocaleString();

export function MyChatsWindow() {
  const {
    sessions,
    activeSessionId,
    ensureActiveSession,
    createSession,
    setActiveSessionId,
    renameSession,
    archiveSession,
    deleteSession
  } = useChatStore();
  const { openWindow } = useOsStore();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [nameDraft, setNameDraft] = useState('');
  const [exportStatus, setExportStatus] = useState<'idle' | 'exporting'>('idle');

  useEffect(() => {
    ensureActiveSession();
  }, [ensureActiveSession]);

  const orderedSessions = useMemo(
    () => [...sessions].sort((a, b) => b.updatedAt - a.updatedAt),
    [sessions]
  );

  const handleOpen = (id: string) => {
    setActiveSessionId(id);
    openWindow('chat');
  };

  const handleRenameStart = (id: string, name: string) => {
    setEditingId(id);
    setNameDraft(name);
  };

  const handleRenameSave = (id: string) => {
    renameSession(id, nameDraft);
    setEditingId(null);
    setNameDraft('');
  };

  const handleArchiveToggle = (id: string, archived: boolean) => {
    archiveSession(id, !archived);
  };

  const handleDelete = (id: string) => {
    if (window.confirm('Delete this chat session? This cannot be undone.')) {
      deleteSession(id);
    }
  };

  const handleExport = async () => {
    setExportStatus('exporting');
    try {
      const zip = new JSZip();
      zip.file('mya-chats.json', JSON.stringify({ sessions }, null, 2));
      const blob = await zip.generateAsync({ type: 'blob' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'mya-chats.zip';
      link.click();
      URL.revokeObjectURL(url);
    } finally {
      setExportStatus('idle');
    }
  };

  return (
    <div className="flex min-h-full flex-col gap-4 text-sm text-retro-text">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold">My Chats</h2>
          <p className="text-xs text-retro-accent">
            Manage, archive, and export your chat sessions.
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              createSession();
              openWindow('chat');
            }}
          >
            New Chat
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => void handleExport()}
            disabled={exportStatus === 'exporting'}
          >
            {exportStatus === 'exporting' ? 'Exporting…' : 'Export'}
          </Button>
        </div>
      </div>

      <div className="flex flex-1 flex-col gap-2 border border-retro-border bg-retro-title-active p-3">
        <div className="text-xs font-semibold">Chat sessions</div>
        {orderedSessions.length === 0 ? (
          <div className="text-xs text-retro-accent">
            No sessions yet. Start a chat to create your first session.
          </div>
        ) : (
          <div className="space-y-3">
            {orderedSessions.map((session) => {
              const isEditing = editingId === session.id;
              const isActive = session.id === activeSessionId;
              return (
                <div
                  key={session.id}
                  className={cn(
                    'flex flex-col gap-2 border border-retro-border bg-retro-surface p-2 text-xs',
                    isActive && 'ring-1 ring-retro-accent'
                  )}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="space-y-1">
                      {isEditing ? (
                        <div className="flex gap-2">
                          <Input
                            value={nameDraft}
                            onChange={(event) => setNameDraft(event.target.value)}
                            className="h-7 text-xs"
                          />
                          <Button
                            size="sm"
                            className="h-7 px-2 text-[11px]"
                            onClick={() => handleRenameSave(session.id)}
                          >
                            Save
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-7 px-2 text-[11px]"
                            onClick={() => {
                              setEditingId(null);
                              setNameDraft('');
                            }}
                          >
                            Cancel
                          </Button>
                        </div>
                      ) : (
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-retro-text">
                            {session.name}
                          </span>
                          {session.archived && (
                            <span className="rounded border border-retro-border px-1 text-[10px] uppercase text-retro-accent">
                              Archived
                            </span>
                          )}
                        </div>
                      )}
                      <div className="text-[11px] text-retro-accent">
                        {session.messages.length} messages · Updated{' '}
                        {formatTimestamp(session.updatedAt)}
                      </div>
                    </div>
                    {!isEditing && (
                      <Button
                        size="sm"
                        className="h-7 px-2 text-[11px]"
                        onClick={() => handleOpen(session.id)}
                      >
                        Open
                      </Button>
                    )}
                  </div>
                  {!isEditing && (
                    <div className="flex flex-wrap gap-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 px-2 text-[11px]"
                        onClick={() => handleRenameStart(session.id, session.name)}
                      >
                        Rename
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 px-2 text-[11px]"
                        onClick={() => handleArchiveToggle(session.id, session.archived)}
                      >
                        {session.archived ? 'Unarchive' : 'Archive'}
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 px-2 text-[11px] text-red-200"
                        onClick={() => handleDelete(session.id)}
                      >
                        Delete
                      </Button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

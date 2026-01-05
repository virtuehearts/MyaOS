'use client';

import { useEffect, useRef, useState } from 'react';

import { Button } from '@/components/ui/button';
import { useAutoScrollToBottom } from '@/components/os/useAutoScrollToBottom';
import { apiRequest } from '@/lib/api';
import { useAuthStore } from '@/store/authStore';
import { useChatStore } from '@/store/chatStore';

type MemoryMetadata = {
  memory_id: string;
  created_at: string;
  tags: string[];
  source_tags: string[];
  virtue_markers: Record<string, number>;
  salience: number;
  updated_at?: string | null;
};

type MemoryRecord = {
  metadata: MemoryMetadata;
  content: string;
};

type MemoryListResponse = {
  memories: MemoryRecord[];
};

type MemoryCreateResponse = MemoryRecord;

type MemoryDeleteResponse = {
  status: string;
  memory_id: string;
};

const toUiEntry = (record: MemoryRecord) => ({
  id: record.metadata.memory_id,
  content: record.content,
  createdAt: new Date(record.metadata.created_at).getTime()
});

export function MemoryWindow() {
  const {
    memory,
    useMemory,
    setUseMemory,
    setMemory,
    addMemory,
    removeMemory,
    clearMemory
  } = useChatStore();
  const { token, user } = useAuthStore();
  const [draft, setDraft] = useState('');
  const [status, setStatus] = useState<'idle' | 'loading' | 'saving' | 'error'>(
    'idle'
  );
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useAutoScrollToBottom(bottomRef, [memory.length]);

  const hasSession = Boolean(token && user);

  useEffect(() => {
    if (!hasSession) {
      setError(null);
      return;
    }

    let isMounted = true;
    const fetchMemories = async () => {
      setStatus('loading');
      setError(null);
      try {
        const response = await apiRequest<MemoryListResponse>('/memory', {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (!isMounted) {
          return;
        }
        setMemory(response.memories.map(toUiEntry));
        setStatus('idle');
      } catch (err) {
        if (!isMounted) {
          return;
        }
        setStatus('error');
        setError(err instanceof Error ? err.message : 'Failed to load memories.');
      }
    };

    fetchMemories();
    return () => {
      isMounted = false;
    };
  }, [hasSession, setMemory, token]);

  const handleAdd = async () => {
    const trimmed = draft.trim();
    if (!trimmed) {
      return;
    }
    if (!hasSession) {
      setError('Sign in to store memories in the vault.');
      return;
    }

    setStatus('saving');
    setError(null);
    try {
      const record = await apiRequest<MemoryCreateResponse>('/memory', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: JSON.stringify({
          content: trimmed,
          tags: ['manual'],
          source_tags: ['vault'],
          virtue_markers: {},
          salience: 0.65
        })
      });
      addMemory(toUiEntry(record));
      setDraft('');
      setStatus('idle');
    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : 'Failed to save memory.');
    }
  };

  const handleRemove = async (memoryId: string) => {
    if (!hasSession) {
      setError('Sign in to manage stored memories.');
      return;
    }

    setStatus('saving');
    setError(null);
    try {
      await apiRequest<MemoryDeleteResponse>(`/memory/${memoryId}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` }
      });
      removeMemory(memoryId);
      setStatus('idle');
    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : 'Failed to remove memory.');
    }
  };

  const handleClearAll = async () => {
    if (!hasSession) {
      setError('Sign in to manage stored memories.');
      return;
    }

    setStatus('saving');
    setError(null);
    try {
      await apiRequest('/memory/import/json?replace=true', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: JSON.stringify({
          memories: [],
          emotion_state: null,
          emotion_baseline: null,
          emotion_transitions: []
        })
      });
      clearMemory();
      setStatus('idle');
    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : 'Failed to clear memories.');
    }
  };

  return (
    <div className="flex min-h-full flex-col gap-4 text-sm text-retro-text">
      <div>
        <h2 className="text-base font-semibold">Memory Vault</h2>
        <p className="text-xs text-retro-accent">
          Store key facts or reminders that Mya can reference later.
        </p>
        {!hasSession && (
          <p className="mt-2 text-xs text-amber-200">
            Sign in to sync memories with the vault. Local-only memories do not persist
            across sessions.
          </p>
        )}
        {status === 'loading' && (
          <p className="mt-2 text-xs text-retro-accent">Loading memories…</p>
        )}
        {error && <p className="mt-2 text-xs text-red-200">{error}</p>}
      </div>

      <label className="flex items-center gap-2 text-xs text-retro-accent">
        <input
          type="checkbox"
          checked={useMemory}
          onChange={(event) => setUseMemory(event.target.checked)}
        />
        Include memories when chatting
      </label>

      <div className="flex flex-col gap-2 border border-retro-border bg-retro-title-active p-3">
        <div className="text-xs font-semibold">Add a memory</div>
        <textarea
          className="min-h-[72px] w-full resize-none border border-retro-border bg-retro-surface p-2 text-xs text-retro-text"
          placeholder="e.g. I prefer concise answers and morning reminders."
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
        />
        <div className="flex justify-end gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setDraft('')}
            className="text-xs"
            disabled={status === 'saving'}
          >
            Clear
          </Button>
          <Button
            size="sm"
            onClick={handleAdd}
            className="text-xs"
            disabled={status === 'saving'}
          >
            Save Memory
          </Button>
        </div>
      </div>

      <div className="flex flex-1 flex-col gap-2 border border-retro-border bg-retro-title-active p-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold">Stored memories</span>
          {memory.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClearAll}
              className="text-xs"
              disabled={status === 'saving'}
            >
              Clear All
            </Button>
          )}
        </div>
        {memory.length === 0 ? (
          <div className="text-xs text-retro-accent">
            No memories yet. Start a chat and save what matters.
          </div>
        ) : (
          <div className="space-y-3">
            {memory.map((item) => (
              <div
                key={item.id}
                className="flex items-start justify-between gap-3 border border-retro-border bg-retro-surface p-2 text-xs"
              >
                <span className="text-retro-text">{item.content}</span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleRemove(item.id)}
                  className="h-6 px-2 text-[10px]"
                  disabled={status === 'saving'}
                >
                  Remove
                </Button>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </div>
    </div>
  );
}

'use client';

import { useRef, useState } from 'react';

import { Button } from '@/components/ui/button';
import { useAutoScrollToBottom } from '@/components/os/useAutoScrollToBottom';
import { useChatStore } from '@/store/chatStore';

const makeId = () =>
  typeof crypto !== 'undefined' ? crypto.randomUUID() : `${Date.now()}`;

export function MemoryWindow() {
  const { memory, useMemory, setUseMemory, addMemory, removeMemory, clearMemory } =
    useChatStore();
  const [draft, setDraft] = useState('');
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useAutoScrollToBottom(bottomRef, [memory.length]);

  const handleAdd = () => {
    const trimmed = draft.trim();
    if (!trimmed) {
      return;
    }
    addMemory({ id: makeId(), content: trimmed, createdAt: Date.now() });
    setDraft('');
  };

  return (
    <div className="flex min-h-full flex-col gap-4 text-sm text-retro-text">
      <div>
        <h2 className="text-base font-semibold">Memory Vault</h2>
        <p className="text-xs text-retro-accent">
          Store key facts or reminders that Mya can reference later.
        </p>
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
          >
            Clear
          </Button>
          <Button size="sm" onClick={handleAdd} className="text-xs">
            Save Memory
          </Button>
        </div>
      </div>

      <div className="flex flex-1 flex-col gap-2 border border-retro-border bg-retro-title-active p-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold">Stored memories</span>
          {memory.length > 0 && (
            <Button variant="ghost" size="sm" onClick={clearMemory} className="text-xs">
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
                  onClick={() => removeMemory(item.id)}
                  className="h-6 px-2 text-[10px]"
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

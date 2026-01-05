'use client';

import { useMemo, useRef, useState } from 'react';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { useAutoScrollToBottom } from '@/components/os/useAutoScrollToBottom';
import { useChatStore } from '@/store/chatStore';
import { useEmotionStore } from '@/store/emotionStore';

type ConsoleEntry = {
  id: string;
  type: 'command' | 'output' | 'error' | 'system';
  content: string;
};

const makeId = () =>
  typeof crypto !== 'undefined' ? crypto.randomUUID() : `${Date.now()}`;

const initialEntries: ConsoleEntry[] = [
  { id: 'boot-1', type: 'system', content: 'MyaOS Console v1.0' },
  {
    id: 'boot-2',
    type: 'output',
    content: 'Type "help" for a list of commands.'
  }
];

const formatPrompt = (path: string) => `myaos:${path}$`;

export function ConsoleWindow() {
  const { memory, clearMemory } = useChatStore();
  const { emotion, setEmotion } = useEmotionStore();
  const [currentPath, setCurrentPath] = useState('/');
  const [draft, setDraft] = useState('');
  const [entries, setEntries] = useState<ConsoleEntry[]>(initialEntries);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useAutoScrollToBottom(bottomRef, [entries]);

  const availableDirectories = useMemo(
    () => ({
      '/': ['apps', 'memory', 'system'],
      '/apps': ['chat', 'memory', 'settings', 'console'],
      '/memory': [`vault.txt (${memory.length} entries)`],
      '/system': ['status.log', 'boot.cfg']
    }),
    [memory.length]
  );

  const resolvePath = (path: string) => {
    if (path === '..') {
      return '/';
    }
    if (path.startsWith('/')) {
      return path;
    }
    return currentPath === '/' ? `/${path}` : `${currentPath}/${path}`;
  };

  const appendEntries = (newEntries: ConsoleEntry[]) => {
    setEntries((prev) => [...prev, ...newEntries]);
  };

  const handleCommand = () => {
    const trimmed = draft.trim();
    if (!trimmed) {
      return;
    }

    const [command, ...args] = trimmed.split(/\s+/);
    const outputEntries: ConsoleEntry[] = [];

    const prompt = `${formatPrompt(currentPath)} ${trimmed}`;
    outputEntries.push({ id: makeId(), type: 'command', content: prompt });

    switch (command) {
      case 'help':
        outputEntries.push({
          id: makeId(),
          type: 'output',
          content:
            'Commands: ls, cd <dir>, set_emotion <state>, clear_memories, status, clear'
        });
        break;
      case 'ls': {
        const listing = availableDirectories[currentPath] ?? [];
        outputEntries.push({
          id: makeId(),
          type: 'output',
          content: listing.length > 0 ? listing.join('  ') : 'Directory empty.'
        });
        break;
      }
      case 'cd': {
        if (args.length === 0) {
          outputEntries.push({
            id: makeId(),
            type: 'error',
            content: 'Usage: cd <dir>'
          });
          break;
        }
        const nextPath = resolvePath(args[0]);
        if (!(nextPath in availableDirectories)) {
          outputEntries.push({
            id: makeId(),
            type: 'error',
            content: `Directory not found: ${args[0]}`
          });
          break;
        }
        setCurrentPath(nextPath);
        outputEntries.push({
          id: makeId(),
          type: 'output',
          content: `Current directory: ${nextPath}`
        });
        break;
      }
      case 'set_emotion': {
        if (args.length === 0) {
          outputEntries.push({
            id: makeId(),
            type: 'error',
            content: 'Usage: set_emotion <state>'
          });
          break;
        }
        const nextEmotion = args.join(' ');
        setEmotion(nextEmotion);
        outputEntries.push({
          id: makeId(),
          type: 'output',
          content: `Emotion set to "${nextEmotion}".`
        });
        break;
      }
      case 'clear_memories':
        clearMemory();
        outputEntries.push({
          id: makeId(),
          type: 'output',
          content: 'Memory vault cleared.'
        });
        break;
      case 'status':
        outputEntries.push({
          id: makeId(),
          type: 'output',
          content: `emotion=${emotion} · memories=${memory.length} · cwd=${currentPath}`
        });
        break;
      case 'clear':
        setEntries(initialEntries);
        setDraft('');
        return;
      default:
        outputEntries.push({
          id: makeId(),
          type: 'error',
          content: `Unknown command: ${command}`
        });
        break;
    }

    appendEntries(outputEntries);
    setDraft('');
  };

  return (
    <Card className="flex min-h-full flex-col border border-retro-border bg-retro-surface text-retro-text">
      <CardHeader className="border-b border-retro-border bg-retro-title-active">
        <CardTitle className="text-sm font-semibold text-retro-text">Console</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col gap-4 text-xs text-retro-accent">
        <div className="space-y-2 font-mono text-[11px]">
          {entries.map((entry) => (
            <div
              key={entry.id}
              className={
                entry.type === 'error'
                  ? 'text-red-200'
                  : entry.type === 'command'
                    ? 'text-retro-text'
                    : entry.type === 'system'
                      ? 'text-retro-text/80'
                      : 'text-retro-accent'
              }
            >
              {entry.content}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
        <form
          className="flex items-center gap-2 border-t border-retro-border pt-2 font-mono text-[11px]"
          onSubmit={(event) => {
            event.preventDefault();
            handleCommand();
          }}
        >
          <span className="text-retro-text">{formatPrompt(currentPath)}</span>
          <Input
            className="h-7 flex-1 bg-retro-title-active text-[11px]"
            placeholder="Type a command…"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
          />
        </form>
      </CardContent>
    </Card>
  );
}

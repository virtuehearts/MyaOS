'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { Send } from 'lucide-react';

import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { createOpenRouterChatCompletion } from '@/lib/openrouter';
import { useApiKeyStore } from '@/store/apiKeyStore';
import { useChatStore } from '@/store/chatStore';

const makeId = () =>
  typeof crypto !== 'undefined' ? crypto.randomUUID() : `${Date.now()}`;

export function ChatWindow() {
  const { apiKey } = useApiKeyStore();
  const {
    messages,
    memory,
    model,
    temperature,
    useMemory,
    addMessage,
    clearMessages
  } = useChatStore();
  const [draft, setDraft] = useState('');
  const [status, setStatus] = useState<'idle' | 'sending' | 'error'>('idle');
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const hasApiKey = Boolean(apiKey);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, status]);

  const systemMemory = useMemo(() => {
    if (!useMemory || memory.length === 0) {
      return null;
    }
    return memory.map((item) => `• ${item.content}`).join('\n');
  }, [memory, useMemory]);

  const handleSend = async () => {
    const trimmed = draft.trim();
    if (!trimmed || status === 'sending') {
      return;
    }

    setError(null);
    setStatus('sending');
    const userMessage = {
      id: makeId(),
      role: 'user' as const,
      content: trimmed,
      createdAt: Date.now()
    };
    addMessage(userMessage);
    setDraft('');

    try {
      const openRouterMessages = [
        ...(systemMemory
          ? [
              {
                role: 'system' as const,
                content: `Memory context:\n${systemMemory}`
              }
            ]
          : []),
        ...messages.map((message) => ({
          role: message.role,
          content: message.content
        })),
        { role: 'user' as const, content: trimmed }
      ];

      const response = await createOpenRouterChatCompletion({
        model,
        temperature,
        messages: openRouterMessages
      });

      addMessage({
        id: makeId(),
        role: 'assistant',
        content: response || '...thinking...',
        createdAt: Date.now()
      });
      setStatus('idle');
    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : 'Unable to reach OpenRouter.');
    }
  };

  return (
    <Card className="flex h-full flex-col border border-retro-border bg-retro-surface text-retro-text">
      <CardHeader className="border-b border-retro-border bg-retro-title-active">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-sm font-semibold text-retro-text">Mya Chat</CardTitle>
          <Button
            variant="ghost"
            size="sm"
            onClick={clearMessages}
            className="h-7 px-2 text-xs"
          >
            Clear Chat
          </Button>
        </div>
        <p className="text-xs text-retro-accent">
          Model: {model} · Temp: {temperature.toFixed(2)}
        </p>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col gap-4">
        {!hasApiKey && (
          <div className="rounded border border-retro-border bg-retro-title-active px-3 py-2 text-xs text-retro-accent">
            Add your OpenRouter API key in Settings to start chatting.
          </div>
        )}
        <ScrollArea className="flex-1 pr-2">
          <div className="space-y-4">
            {messages.length === 0 && (
              <div className="text-xs text-retro-accent">
                No messages yet. Say hello to begin.
              </div>
            )}
            {messages.map((message) => (
              <div key={message.id} className="flex items-start gap-3">
                <Avatar className="h-9 w-9">
                  <AvatarFallback>
                    {message.role === 'assistant' ? 'M' : 'Y'}
                  </AvatarFallback>
                </Avatar>
                <div>
                  <p className="text-sm font-semibold text-retro-text">
                    {message.role === 'assistant' ? 'Mya' : 'You'}
                  </p>
                  <p className="text-sm text-retro-accent">{message.content}</p>
                </div>
              </div>
            ))}
            {status === 'sending' && (
              <div className="text-xs text-retro-accent">Mya is thinking…</div>
            )}
            {error && <div className="text-xs text-red-200">Error: {error}</div>}
            <div ref={bottomRef} />
          </div>
        </ScrollArea>
        <div className="flex gap-2">
          <Input
            placeholder="Share your thoughts with Mya..."
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                void handleSend();
              }
            }}
            disabled={!hasApiKey}
          />
          <Button
            className="px-3"
            aria-label="Send message"
            onClick={() => void handleSend()}
            disabled={!draft.trim() || status === 'sending' || !hasApiKey}
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

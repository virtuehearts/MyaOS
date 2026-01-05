'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { Copy, RefreshCw, Send } from 'lucide-react';

import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { useAutoScrollToBottom } from '@/components/os/useAutoScrollToBottom';
import type { OpenRouterChatOptions } from '@/lib/openrouter';
import { createOpenRouterChatCompletion } from '@/lib/openrouter';
import { useApiKeyStore } from '@/store/apiKeyStore';
import { useChatStore } from '@/store/chatStore';
import { useEmotionStore } from '@/store/emotionStore';

const makeId = () =>
  typeof crypto !== 'undefined' ? crypto.randomUUID() : `${Date.now()}`;

const CONTROL_BLOCK_START = '<control>';
const CONTROL_BLOCK_END = '</control>';

const stripControlBlock = (content: string) => {
  const start = content.indexOf(CONTROL_BLOCK_START);
  const end = content.indexOf(CONTROL_BLOCK_END);
  if (start === -1 || end === -1 || end < start) {
    return { cleaned: content.trim(), control: null };
  }
  const control = content.slice(start + CONTROL_BLOCK_START.length, end).trim();
  const cleaned = `${content.slice(0, start)}${content.slice(
    end + CONTROL_BLOCK_END.length
  )}`.trim();
  return { cleaned, control };
};

export function ChatWindow() {
  const { apiKey } = useApiKeyStore();
  const {
    sessions,
    activeSessionId,
    memory,
    model,
    temperature,
    useMemory,
    persona,
    usePersona,
    ensureActiveSession,
    addMessage,
    clearMessages
  } = useChatStore();
  const { emotion } = useEmotionStore();
  const [draft, setDraft] = useState('');
  const [status, setStatus] = useState<'idle' | 'sending' | 'error'>('idle');
  const [error, setError] = useState<string | null>(null);
  const [lastRequest, setLastRequest] = useState<OpenRouterChatOptions | null>(
    null
  );
  const [controlState, setControlState] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const hasApiKey = Boolean(apiKey);

  useEffect(() => {
    ensureActiveSession();
  }, [ensureActiveSession]);

  const activeSession = useMemo(
    () => sessions.find((session) => session.id === activeSessionId) ?? null,
    [sessions, activeSessionId]
  );

  const messages = activeSession?.messages ?? [];

  useAutoScrollToBottom(bottomRef, [messages, status]);

  const systemMemory = useMemo(() => {
    if (!useMemory || memory.length === 0) {
      return null;
    }
    return memory.map((item) => `• ${item.content}`).join('\n');
  }, [memory, useMemory]);

  const systemPrompt = useMemo(() => {
    const sections: string[] = [];
    const trimmedPersona = persona.trim();
    if (usePersona && trimmedPersona) {
      sections.push(`Persona/character context:\n${trimmedPersona}`);
    }
    if (systemMemory) {
      sections.push(`Memory context:\n${systemMemory}`);
    }
    sections.push(`Current emotional tone: ${emotion}`);
    sections.push(
      [
        'Include control data for emotional-state processing in a <control>...</control> block at the end of your reply.',
        'The control block should contain only control JSON and should never be user-facing content.',
        controlState ? `Previous control data:\n${controlState}` : null
      ]
        .filter(Boolean)
        .join('\n')
    );
    return sections.join('\n\n');
  }, [controlState, emotion, persona, systemMemory, usePersona]);

  const submitRequest = async (request: OpenRouterChatOptions) => {
    setStatus('sending');
    setError(null);

    try {
      const response = await createOpenRouterChatCompletion(request);
      const { cleaned, control } = stripControlBlock(response);

      addMessage({
        id: makeId(),
        role: 'assistant',
        content: cleaned || '...thinking...',
        createdAt: Date.now()
      });
      if (control) {
        setControlState(control);
      }
      setStatus('idle');
      setLastRequest(null);
    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : 'Unable to reach OpenRouter.');
    }
  };

  const handleSend = async () => {
    const trimmed = draft.trim();
    if (!trimmed || status === 'sending') {
      return;
    }

    const userMessage = {
      id: makeId(),
      role: 'user' as const,
      content: trimmed,
      createdAt: Date.now()
    };
    addMessage(userMessage);
    setDraft('');

    const openRouterMessages = [
      {
        role: 'system' as const,
        content: systemPrompt
      },
      ...messages.map((message) => ({
        role: message.role,
        content: message.content
      })),
      { role: 'user' as const, content: trimmed }
    ];

    const request = {
      model,
      temperature,
      messages: openRouterMessages
    };

    setLastRequest(request);
    await submitRequest(request);
  };

  const handleRetry = async () => {
    if (!lastRequest || status === 'sending') {
      return;
    }
    await submitRequest(lastRequest);
  };

  const handleCopy = async (content: string) => {
    try {
      await navigator.clipboard.writeText(content);
    } catch {
      // ignore clipboard failures
    }
  };

  const lastAssistantMessage = [...messages]
    .reverse()
    .find((message) => message.role === 'assistant');

  return (
    <Card className="flex min-h-full flex-col border border-retro-border bg-retro-surface text-retro-text">
      <CardHeader className="border-b border-retro-border bg-retro-title-active">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-sm font-semibold text-retro-text">
            {activeSession?.name ?? 'Mya Chat'}
          </CardTitle>
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
        <div className="relative flex-1">
          <div className="space-y-4 pb-12">
            {messages.length === 0 && (
              <div className="text-xs text-retro-accent">
                No messages yet. Say hello to begin.
              </div>
            )}
            {messages.map((message) => (
              <div
                key={message.id}
                className="rounded-xl border border-retro-border bg-retro-title-active/70 p-3"
              >
                <div className="flex items-start gap-3">
                  <Avatar className="h-9 w-9">
                    <AvatarFallback>
                      {message.role === 'assistant' ? 'M' : 'Y'}
                    </AvatarFallback>
                  </Avatar>
                  <div className="flex-1 space-y-2">
                    <div>
                      <p className="text-sm font-semibold text-retro-text">
                        {message.role === 'assistant' ? 'Mya' : 'You'}
                      </p>
                      <p className="text-sm text-retro-accent">
                        {message.content}
                      </p>
                    </div>
                    <div className="flex justify-end">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-retro-accent hover:text-retro-text"
                        aria-label="Copy message"
                        onClick={() => void handleCopy(message.content)}
                      >
                        <Copy className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            ))}
            {status === 'sending' && (
              <div className="text-xs text-retro-accent">Mya is thinking…</div>
            )}
            <div ref={bottomRef} />
          </div>
          <div className="absolute bottom-3 right-3 flex items-center gap-1 rounded-full border border-retro-border bg-retro-title-active/90 p-1 shadow-lg">
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-retro-accent hover:text-retro-text"
              aria-label="Regenerate response"
              onClick={() => void handleRetry()}
              disabled={!lastRequest || status === 'sending'}
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-retro-accent hover:text-retro-text"
              aria-label="Copy latest response"
              onClick={() =>
                lastAssistantMessage
                  ? void handleCopy(lastAssistantMessage.content)
                  : undefined
              }
              disabled={!lastAssistantMessage}
            >
              <Copy className="h-4 w-4" />
            </Button>
          </div>
        </div>
        {error && (
          <div className="flex flex-wrap items-center justify-between gap-2 rounded border border-red-300/50 bg-red-950/40 px-3 py-2 text-xs text-red-100">
            <span>Request failed: {error}</span>
            <div className="flex gap-2">
              <Button
                size="sm"
                onClick={() => void handleRetry()}
                disabled={!lastRequest || status === 'sending'}
                className="h-7 px-2 text-[11px]"
              >
                Retry
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setError(null);
                  setStatus('idle');
                }}
                className="h-7 px-2 text-[11px]"
              >
                Dismiss
              </Button>
            </div>
          </div>
        )}
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

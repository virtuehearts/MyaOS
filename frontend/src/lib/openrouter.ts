'use client';

import { useApiKeyStore } from '@/store/apiKeyStore';

export const OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1';
export const DEFAULT_OPENROUTER_MODEL = 'nvidia/nemotron-nano-12b-v2-vl:free';
export const OPENROUTER_CONTEXT_WINDOW_TOKENS = 128000;
export const OPENROUTER_RESERVED_CONTEXT_TOKENS = 32000;
export const OPENROUTER_MAX_COMPLETION_TOKENS =
  OPENROUTER_CONTEXT_WINDOW_TOKENS - OPENROUTER_RESERVED_CONTEXT_TOKENS;

export interface OpenRouterMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface OpenRouterChatOptions {
  model: string;
  temperature: number;
  messages: OpenRouterMessage[];
  maxTokens?: number;
  signal?: AbortSignal;
}

const getStoredKey = () => {
  if (typeof window === 'undefined') {
    return '';
  }
  return window.localStorage.getItem('openrouter_api_key') ?? '';
};

const resolveApiKey = () => {
  const keyFromStore = useApiKeyStore.getState?.().apiKey ?? '';
  return keyFromStore || getStoredKey();
};

export async function createOpenRouterChatCompletion({
  model,
  temperature,
  messages,
  maxTokens,
  signal
}: OpenRouterChatOptions): Promise<string> {
  const apiKey = resolveApiKey();
  if (!apiKey) {
    throw new Error('Missing OpenRouter API key.');
  }

  const referer =
    typeof window === 'undefined' ? 'http://localhost' : window.location.origin;

  const response = await fetch(`${OPENROUTER_BASE_URL}/chat/completions`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'HTTP-Referer': referer,
      'X-Title': 'MyaOS'
    },
    body: JSON.stringify({
      model,
      temperature,
      messages,
      max_tokens:
        maxTokens ??
        (model === DEFAULT_OPENROUTER_MODEL
          ? OPENROUTER_MAX_COMPLETION_TOKENS
          : undefined)
    }),
    signal
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const message =
      payload?.error?.message ??
      payload?.message ??
      `OpenRouter request failed (${response.status}).`;
    throw new Error(message);
  }

  const data = await response.json();
  return data.choices?.[0]?.message?.content?.trim() ?? '';
}

'use client';

import { useApiKeyStore } from '@/store/apiKeyStore';

export const OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1';

export interface OpenRouterMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface OpenRouterChatOptions {
  model: string;
  temperature: number;
  messages: OpenRouterMessage[];
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
      messages
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

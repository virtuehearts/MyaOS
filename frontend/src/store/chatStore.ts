'use client';

import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  createdAt: number;
}

export interface MemoryEntry {
  id: string;
  content: string;
  createdAt: number;
}

interface ChatStore {
  messages: ChatMessage[];
  memory: MemoryEntry[];
  model: string;
  temperature: number;
  useMemory: boolean;
  persona: string;
  usePersona: boolean;
  addMessage: (message: ChatMessage) => void;
  clearMessages: () => void;
  addMemory: (entry: MemoryEntry) => void;
  removeMemory: (id: string) => void;
  clearMemory: () => void;
  setModel: (model: string) => void;
  setTemperature: (temperature: number) => void;
  setUseMemory: (value: boolean) => void;
  setPersona: (persona: string) => void;
  setUsePersona: (value: boolean) => void;
}

const storage =
  typeof window === 'undefined'
    ? undefined
    : createJSONStorage(() => window.localStorage);

export const useChatStore = create<ChatStore>()(
  persist(
    (set) => ({
      messages: [],
      memory: [],
      model: 'nvidia/nemotron-nano-12b-v2-vl:free',
      temperature: 0.7,
      useMemory: true,
      persona: '',
      usePersona: false,
      addMessage: (message) =>
        set((state) => ({ messages: [...state.messages, message] })),
      clearMessages: () => set({ messages: [] }),
      addMemory: (entry) =>
        set((state) => ({ memory: [entry, ...state.memory] })),
      removeMemory: (id) =>
        set((state) => ({ memory: state.memory.filter((item) => item.id !== id) })),
      clearMemory: () => set({ memory: [] }),
      setModel: (model) => set({ model }),
      setTemperature: (temperature) => set({ temperature }),
      setUseMemory: (value) => set({ useMemory: value }),
      setPersona: (persona) => set({ persona }),
      setUsePersona: (value) => set({ usePersona: value })
    }),
    {
      name: 'mya-chat-store',
      storage,
      partialize: (state) => ({
        messages: state.messages,
        memory: state.memory,
        model: state.model,
        temperature: state.temperature,
        useMemory: state.useMemory,
        persona: state.persona,
        usePersona: state.usePersona
      })
    }
  )
);

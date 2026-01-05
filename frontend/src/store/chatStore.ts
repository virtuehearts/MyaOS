'use client';

import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  createdAt: number;
}

export interface ChatSession {
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
  archived: boolean;
}

export interface MemoryEntry {
  id: string;
  content: string;
  createdAt: number;
}

interface ChatStore {
  sessions: ChatSession[];
  activeSessionId: string | null;
  shouldStartNewSessionOnOpen: boolean;
  memory: MemoryEntry[];
  model: string;
  temperature: number;
  useMemory: boolean;
  persona: string;
  usePersona: boolean;
  ensureActiveSession: () => void;
  createSession: (name?: string) => void;
  setActiveSessionId: (id: string) => void;
  renameSession: (id: string, name: string) => void;
  archiveSession: (id: string, archived: boolean) => void;
  deleteSession: (id: string) => void;
  markChatClosed: () => void;
  addMessage: (message: ChatMessage) => void;
  clearMessages: () => void;
  setMemory: (entries: MemoryEntry[]) => void;
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

const makeId = () =>
  typeof crypto !== 'undefined' ? crypto.randomUUID() : `${Date.now()}`;

const createSessionTemplate = (name: string) => {
  const now = Date.now();
  return {
    id: makeId(),
    name,
    createdAt: now,
    updatedAt: now,
    messages: [],
    archived: false
  } satisfies ChatSession;
};

const ensureUniqueName = (sessions: ChatSession[]) =>
  `Chat ${sessions.length + 1}`;

const initialSession = createSessionTemplate('Chat 1');

export const useChatStore = create<ChatStore>()(
  persist(
    (set) => ({
      sessions: [initialSession],
      activeSessionId: initialSession.id,
      shouldStartNewSessionOnOpen: false,
      memory: [],
      model: 'nvidia/nemotron-nano-12b-v2-vl:free',
      temperature: 0.7,
      useMemory: true,
      persona: '',
      usePersona: false,
      ensureActiveSession: () =>
        set((state) => {
          if (state.sessions.length === 0 || state.shouldStartNewSessionOnOpen) {
            const session = createSessionTemplate(ensureUniqueName(state.sessions));
            return {
              sessions: [session, ...state.sessions],
              activeSessionId: session.id,
              shouldStartNewSessionOnOpen: false
            };
          }

          if (!state.activeSessionId) {
            return { activeSessionId: state.sessions[0]?.id ?? null };
          }

          const hasActive = state.sessions.some(
            (session) => session.id === state.activeSessionId
          );
          if (!hasActive) {
            return { activeSessionId: state.sessions[0]?.id ?? null };
          }
          return {};
        }),
      createSession: (name) =>
        set((state) => {
          const session = createSessionTemplate(
            name?.trim() || ensureUniqueName(state.sessions)
          );
          return {
            sessions: [session, ...state.sessions],
            activeSessionId: session.id,
            shouldStartNewSessionOnOpen: false
          };
        }),
      setActiveSessionId: (id) =>
        set((state) => ({
          activeSessionId: state.sessions.some((session) => session.id === id)
            ? id
            : state.activeSessionId,
          shouldStartNewSessionOnOpen: false
        })),
      renameSession: (id, name) =>
        set((state) => ({
          sessions: state.sessions.map((session) =>
            session.id === id
              ? {
                  ...session,
                  name: name.trim() || session.name,
                  updatedAt: Date.now()
                }
              : session
          )
        })),
      archiveSession: (id, archived) =>
        set((state) => ({
          sessions: state.sessions.map((session) =>
            session.id === id
              ? {
                  ...session,
                  archived,
                  updatedAt: Date.now()
                }
              : session
          )
        })),
      deleteSession: (id) =>
        set((state) => {
          const nextSessions = state.sessions.filter((session) => session.id !== id);
          const shouldResetActive = state.activeSessionId === id;
          return {
            sessions: nextSessions,
            activeSessionId: shouldResetActive
              ? nextSessions[0]?.id ?? null
              : state.activeSessionId
          };
        }),
      markChatClosed: () => set({ shouldStartNewSessionOnOpen: true }),
      addMessage: (message) =>
        set((state) => ({
          sessions: state.sessions.map((session) =>
            session.id === state.activeSessionId
              ? {
                  ...session,
                  messages: [...session.messages, message],
                  updatedAt: Date.now()
                }
              : session
          )
        })),
      clearMessages: () =>
        set((state) => ({
          sessions: state.sessions.map((session) =>
            session.id === state.activeSessionId
              ? { ...session, messages: [], updatedAt: Date.now() }
              : session
          )
        })),
      setMemory: (entries) => set({ memory: entries }),
      addMemory: (entry) =>
        set((state) => {
          if (state.memory.some((item) => item.id === entry.id)) {
            return { memory: state.memory };
          }
          return { memory: [entry, ...state.memory] };
        }),
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
      version: 1,
      migrate: (persistedState, version) => {
        if (version === 0 && persistedState && typeof persistedState === 'object') {
          const legacyState = persistedState as {
            messages?: ChatMessage[];
            memory?: MemoryEntry[];
            model?: string;
            temperature?: number;
            useMemory?: boolean;
            persona?: string;
            usePersona?: boolean;
          };
          const legacyMessages = legacyState.messages ?? [];
          const now = Date.now();
          const createdAt = legacyMessages[0]?.createdAt ?? now;
          const updatedAt =
            legacyMessages[legacyMessages.length - 1]?.createdAt ?? createdAt;
          const session: ChatSession = {
            id: makeId(),
            name: 'Chat 1',
            createdAt,
            updatedAt,
            messages: legacyMessages,
            archived: false
          };
          return {
            sessions: [session],
            activeSessionId: session.id,
            shouldStartNewSessionOnOpen: false,
            memory: legacyState.memory ?? [],
            model: legacyState.model ?? 'nvidia/nemotron-nano-12b-v2-vl:free',
            temperature: legacyState.temperature ?? 0.7,
            useMemory: legacyState.useMemory ?? true,
            persona: legacyState.persona ?? '',
            usePersona: legacyState.usePersona ?? false
          };
        }
        return persistedState as ChatStore;
      },
      partialize: (state) => ({
        sessions: state.sessions,
        activeSessionId: state.activeSessionId,
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

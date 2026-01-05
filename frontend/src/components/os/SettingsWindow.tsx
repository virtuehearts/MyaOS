'use client';

import { useEffect, useRef, useState } from 'react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useApiKeyStore } from '@/store/apiKeyStore';
import { useChatStore } from '@/store/chatStore';

interface SettingsWindowProps {
  desktopBackground?: string | null;
  onBackgroundSelect: (file: File) => void;
  onClearBackground: () => void;
}

export function SettingsWindow({
  desktopBackground,
  onBackgroundSelect,
  onClearBackground
}: SettingsWindowProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const { apiKey, setApiKey, clearApiKey } = useApiKeyStore();
  const {
    model,
    temperature,
    useMemory,
    persona,
    usePersona,
    setModel,
    setTemperature,
    setUseMemory,
    setPersona,
    setUsePersona
  } = useChatStore();
  const [localApiKey, setLocalApiKey] = useState(apiKey);
  const [localModel, setLocalModel] = useState(model);
  const [localTemperature, setLocalTemperature] = useState(temperature.toString());
  const [localPersona, setLocalPersona] = useState(persona);
  const [localUsePersona, setLocalUsePersona] = useState(usePersona);
  const [localUseMemory, setLocalUseMemory] = useState(useMemory);

  useEffect(() => {
    setLocalApiKey(apiKey);
  }, [apiKey]);

  useEffect(() => {
    setLocalModel(model);
  }, [model]);

  useEffect(() => {
    setLocalTemperature(temperature.toString());
  }, [temperature]);

  useEffect(() => {
    setLocalPersona(persona);
  }, [persona]);

  useEffect(() => {
    setLocalUsePersona(usePersona);
  }, [usePersona]);

  useEffect(() => {
    setLocalUseMemory(useMemory);
  }, [useMemory]);

  return (
    <div className="flex h-full flex-col gap-4 text-sm text-retro-text">
      <div>
        <h2 className="text-base font-semibold">OpenRouter Settings</h2>
        <p className="text-xs text-retro-accent">
          Update your API key and model preferences for Mya.
        </p>
      </div>

      <div className="flex flex-col gap-3 border border-retro-border bg-retro-title-active p-3">
        <label className="text-xs font-semibold">API Key</label>
        <div className="flex items-center gap-2">
          <Input
            type="password"
            placeholder="sk-or-..."
            value={localApiKey}
            onChange={(event) => setLocalApiKey(event.target.value)}
          />
          <Button
            size="sm"
            onClick={() => setApiKey(localApiKey.trim())}
            className="px-3 text-xs"
          >
            Save
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={clearApiKey}
            className="px-3 text-xs"
          >
            Clear
          </Button>
        </div>
        <p className="text-[11px] text-retro-accent">
          Saved locally and never sent anywhere except OpenRouter.
        </p>
      </div>

      <div className="flex flex-col gap-3 border border-retro-border bg-retro-title-active p-3">
        <label className="text-xs font-semibold">Model</label>
        <Input
          placeholder="openai/gpt-4o-mini"
          value={localModel}
          onChange={(event) => setLocalModel(event.target.value)}
          onBlur={() => setModel(localModel.trim() || model)}
        />
        <label className="text-xs font-semibold">Temperature</label>
        <Input
          type="number"
          min="0"
          max="2"
          step="0.1"
          value={localTemperature}
          onChange={(event) => setLocalTemperature(event.target.value)}
          onBlur={() => {
            const nextValue = Number.parseFloat(localTemperature);
            if (!Number.isNaN(nextValue)) {
              setTemperature(nextValue);
            } else {
              setLocalTemperature(temperature.toString());
            }
          }}
        />
        <p className="text-[11px] text-retro-accent">
          Lower values are more focused; higher values are more creative.
        </p>
      </div>

      <div className="flex flex-col gap-3 border border-retro-border bg-retro-title-active p-3">
        <label className="text-xs font-semibold">Persona / Character</label>
        <Input
          placeholder="Describe Mya's persona or role..."
          value={localPersona}
          onChange={(event) => setLocalPersona(event.target.value)}
          onBlur={() => setPersona(localPersona.trim())}
        />
        <label className="flex items-center gap-2 text-xs text-retro-accent">
          <input
            type="checkbox"
            checked={localUsePersona}
            onChange={(event) => {
              const nextValue = event.target.checked;
              setLocalUsePersona(nextValue);
              setUsePersona(nextValue);
            }}
          />
          Include persona in system prompt
        </label>
        <p className="text-[11px] text-retro-accent">
          Enable this to prepend the persona to each chat request.
        </p>
      </div>

      <div>
        <h2 className="text-base font-semibold">Privacy &amp; Memory</h2>
        <p className="text-xs text-retro-accent">
          Memory is stored locally in browser storage. LLM requests go to OpenRouter
          using the API key you provide.
        </p>
      </div>

      <div className="flex flex-col gap-3 border border-retro-border bg-retro-title-active p-3">
        <label className="flex items-center gap-2 text-xs text-retro-accent">
          <input
            type="checkbox"
            checked={localUseMemory}
            onChange={(event) => {
              const nextValue = event.target.checked;
              setLocalUseMemory(nextValue);
              setUseMemory(nextValue);
            }}
          />
          Send memory with chat requests
        </label>
        <p className="text-[11px] text-retro-accent">
          Turn this off to keep saved memory out of prompts sent to the LLM.
        </p>
        <label className="flex items-center gap-2 text-xs text-retro-accent opacity-60">
          <input type="checkbox" disabled />
          Local LLM mode (coming soon)
        </label>
        <p className="text-[11px] text-retro-accent">
          This toggle will be enabled once the local model backend is available.
        </p>
      </div>

      <div>
        <h2 className="text-base font-semibold">Desktop Settings</h2>
        <p className="text-xs text-retro-accent">
          Choose a JPG/PNG wallpaper. Images are auto-stretched to fill the desktop.
        </p>
      </div>

      <div className="flex flex-col gap-3 border border-retro-border bg-retro-title-active p-3">
        <div className="flex items-center justify-between gap-3">
          <div className="text-xs">
            {desktopBackground ? 'Custom wallpaper selected.' : 'No custom wallpaper set.'}
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => fileInputRef.current?.click()}
            >
              Choose Image
            </Button>
            <Button variant="ghost" size="sm" onClick={onClearBackground}>
              Clear
            </Button>
          </div>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/png,image/jpeg"
          className="hidden"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) {
              onBackgroundSelect(file);
            }
            event.target.value = '';
          }}
        />
      </div>
    </div>
  );
}

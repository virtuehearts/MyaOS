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
  const { model, temperature, setModel, setTemperature } = useChatStore();
  const [localApiKey, setLocalApiKey] = useState(apiKey);
  const [localModel, setLocalModel] = useState(model);
  const [localTemperature, setLocalTemperature] = useState(temperature.toString());

  useEffect(() => {
    setLocalApiKey(apiKey);
  }, [apiKey]);

  useEffect(() => {
    setLocalModel(model);
  }, [model]);

  useEffect(() => {
    setLocalTemperature(temperature.toString());
  }, [temperature]);

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

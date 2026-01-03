'use client';

import { useRef } from 'react';

import { Button } from '@/components/ui/button';

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

  return (
    <div className="flex h-full flex-col gap-4 text-sm text-retro-text">
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

'use client';

import { Button } from '@/components/ui/button';

interface TextEditorProps {
  isOpen: boolean;
  filename: string | null;
  content: string;
  isSaving: boolean;
  onChange: (value: string) => void;
  onClose: () => void;
  onSave: () => void;
}

export function TextEditor({
  isOpen,
  filename,
  content,
  isSaving,
  onChange,
  onClose,
  onSave
}: TextEditorProps) {
  if (!isOpen) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/50 p-4 text-sm text-retro-text">
      <div className="w-full max-w-3xl border border-retro-border bg-retro-title-active shadow-[0_12px_0_rgba(0,0,0,0.2)]">
        <div className="flex items-center justify-between border-b border-retro-border px-4 py-2">
          <div>
            <div className="text-xs font-semibold">Text Editor</div>
            <div className="text-[11px] text-retro-accent">
              {filename ?? 'Untitled'}
            </div>
          </div>
          <div className="flex gap-2">
            <Button variant="ghost" size="sm" className="h-7 px-3 text-[10px]" onClick={onClose}>
              Close
            </Button>
            <Button size="sm" className="h-7 px-3 text-[10px]" onClick={onSave} disabled={isSaving}>
              {isSaving ? 'Saving…' : 'Save'}
            </Button>
          </div>
        </div>
        <div className="p-4">
          <textarea
            value={content}
            onChange={(event) => onChange(event.target.value)}
            className="h-[360px] w-full resize-none border border-retro-border bg-retro-surface p-3 text-xs text-retro-text focus:outline-none"
          />
        </div>
      </div>
    </div>
  );
}

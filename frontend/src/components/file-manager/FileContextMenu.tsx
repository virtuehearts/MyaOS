'use client';

import { useEffect, useRef } from 'react';

import { cn } from '@/lib/utils';

export type ContextMenuAction = {
  id: string;
  label: string;
  onSelect: () => void;
  disabled?: boolean;
};

interface FileContextMenuProps {
  isOpen: boolean;
  position: { x: number; y: number };
  actions: ContextMenuAction[];
  onClose: () => void;
}

export function FileContextMenu({
  isOpen,
  position,
  actions,
  onClose
}: FileContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    const handlePointer = (event: MouseEvent) => {
      if (!menuRef.current) {
        return;
      }
      if (!menuRef.current.contains(event.target as Node)) {
        onClose();
      }
    };

    window.addEventListener('mousedown', handlePointer);
    window.addEventListener('contextmenu', handlePointer);
    return () => {
      window.removeEventListener('mousedown', handlePointer);
      window.removeEventListener('contextmenu', handlePointer);
    };
  }, [isOpen, onClose]);

  if (!isOpen) {
    return null;
  }

  return (
    <div
      ref={menuRef}
      className="fixed z-50 min-w-[160px] border border-retro-border bg-retro-title-active text-xs shadow-[0_8px_0_rgba(0,0,0,0.15)]"
      style={{ left: position.x, top: position.y }}
    >
      <div className="flex flex-col py-1">
        {actions.map((action) => (
          <button
            key={action.id}
            type="button"
            onClick={() => {
              if (action.disabled) {
                return;
              }
              action.onSelect();
              onClose();
            }}
            className={cn(
              'px-3 py-1.5 text-left text-retro-text transition hover:bg-retro-surface',
              action.disabled && 'cursor-not-allowed opacity-60 hover:bg-transparent'
            )}
            disabled={action.disabled}
          >
            {action.label}
          </button>
        ))}
      </div>
    </div>
  );
}

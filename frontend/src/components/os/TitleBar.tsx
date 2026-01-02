'use client';

import { X, Minus, Square } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface TitleBarProps {
  title: string;
  isActive?: boolean;
  onClose?: () => void;
  onMinimize?: () => void;
  onMaximize?: () => void;
  variant?: 'mac' | 'windows';
}

export function TitleBar({
  title,
  isActive = false,
  onClose,
  onMinimize,
  onMaximize,
  variant = 'mac'
}: TitleBarProps) {
  return (
    <div
      className={cn(
        'flex items-center justify-between rounded-t-xl border-b border-slate-800/60 px-4 py-2 text-sm',
        isActive ? 'bg-slate-900/70' : 'bg-slate-950/50'
      )}
    >
      <div className="flex items-center gap-2">
        {variant === 'mac' ? (
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              className="h-3 w-3 rounded-full bg-red-500 p-0"
              onClick={onClose}
            />
            <Button
              variant="ghost"
              size="sm"
              className="h-3 w-3 rounded-full bg-yellow-500 p-0"
              onClick={onMinimize}
            />
            <Button
              variant="ghost"
              size="sm"
              className="h-3 w-3 rounded-full bg-green-500 p-0"
              onClick={onMaximize}
            />
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={onMinimize}>
              <Minus className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="sm" onClick={onMaximize}>
              <Square className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="sm" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        )}
      </div>
      <span className="text-slate-200">{title}</span>
      <div className="h-6 w-20" />
    </div>
  );
}

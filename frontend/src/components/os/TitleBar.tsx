'use client';

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
  variant = 'windows'
}: TitleBarProps) {
  const buttonLabelClass = 'h-6 w-6 px-0 text-xs font-bold leading-none border border-retro-border';
  const controls = (
    <>
      <Button
        variant="ghost"
        size="sm"
        className={cn(buttonLabelClass)}
        onClick={onMinimize}
      >
        _
      </Button>
      <Button
        variant="ghost"
        size="sm"
        className={cn(buttonLabelClass)}
        onClick={onMaximize}
      >
        □
      </Button>
      <Button
        variant="ghost"
        size="sm"
        className={cn(buttonLabelClass)}
        onClick={onClose}
      >
        X
      </Button>
    </>
  );

  return (
    <div
      className={cn(
        'os-titlebar-drag flex items-center justify-between border-b border-retro-border px-3 py-1 text-xs',
        isActive ? 'bg-retro-title-active' : 'bg-retro-surface'
      )}
    >
      {variant === 'windows' ? (
        <>
          <span className="text-retro-text">{title}</span>
          <div className="flex items-center gap-1">{controls}</div>
        </>
      ) : (
        <>
          <div className="flex items-center gap-1">{controls}</div>
          <span className="text-retro-text">{title}</span>
          <div className="h-6 w-20" />
        </>
      )}
    </div>
  );
}

'use client';

import { Rnd } from 'react-rnd';

import { TitleBar } from '@/components/os/TitleBar';
import { cn } from '@/lib/utils';

interface OsWindowProps {
  id: string;
  title: string;
  children: React.ReactNode;
  isActive?: boolean;
  isMinimized?: boolean;
  zIndex?: number;
  onClose?: () => void;
  onMinimize?: () => void;
  onFocus?: () => void;
}

export function OsWindow({
  id,
  title,
  children,
  isActive,
  isMinimized,
  zIndex = 10,
  onClose,
  onMinimize,
  onFocus
}: OsWindowProps) {
  if (isMinimized) {
    return null;
  }

  return (
    <Rnd
      default={{
        x: 120,
        y: 120,
        width: 560,
        height: 420
      }}
      minWidth={360}
      minHeight={280}
      bounds="window"
      onDragStart={onFocus}
      onResizeStart={onFocus}
      style={{ zIndex }}
      className="mya-panel rounded-xl"
      aria-label={`${id}-window`}
    >
      <div
        className={cn(
          'flex h-full w-full flex-col overflow-hidden rounded-xl',
          isActive ? 'ring-1 ring-mya-saffron/40' : 'ring-1 ring-transparent'
        )}
      >
        <TitleBar
          title={title}
          isActive={isActive}
          onClose={onClose}
          onMinimize={onMinimize}
        />
        <div className="flex-1 overflow-hidden p-4">{children}</div>
      </div>
    </Rnd>
  );
}

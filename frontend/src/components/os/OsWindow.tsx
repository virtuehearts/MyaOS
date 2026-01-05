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
  width: number;
  height: number;
  minWidth: number;
  minHeight: number;
  zIndex?: number;
  onClose?: () => void;
  onMinimize?: () => void;
  onFocus?: () => void;
  onResizeStart?: () => void;
  onResizeStop?: (size: { width: number; height: number }) => void;
}

export function OsWindow({
  id,
  title,
  children,
  isActive,
  isMinimized,
  width,
  height,
  minWidth,
  minHeight,
  zIndex = 10,
  onClose,
  onMinimize,
  onFocus,
  onResizeStart,
  onResizeStop
}: OsWindowProps) {
  const handleClass =
    'absolute bg-retro-title-active border border-retro-border';

  return (
    <Rnd
      default={{
        x: 120,
        y: 120
      }}
      size={{ width, height }}
      minWidth={minWidth}
      minHeight={minHeight}
      bounds="window"
      onDragStart={onFocus}
      onResizeStart={() => {
        onFocus?.();
        onResizeStart?.();
      }}
      onResizeStop={(_event, _direction, ref) => {
        onResizeStop?.({
          width: ref.offsetWidth,
          height: ref.offsetHeight
        });
      }}
      resizeGrid={[8, 8]}
      style={{ zIndex, display: isMinimized ? 'none' : 'block' }}
      className="mya-panel border"
      aria-label={`${id}-window`}
      resizeHandleComponent={{
        topLeft: (
          <span className={`${handleClass} -left-1 -top-1 h-3 w-3 cursor-nwse-resize`} />
        ),
        top: (
          <span className={`${handleClass} -top-1 left-3 right-3 h-2 cursor-ns-resize`} />
        ),
        topRight: (
          <span className={`${handleClass} -right-1 -top-1 h-3 w-3 cursor-nesw-resize`} />
        ),
        right: (
          <span className={`${handleClass} -right-1 bottom-3 top-3 w-2 cursor-ew-resize`} />
        ),
        bottomRight: (
          <span className={`${handleClass} -bottom-1 -right-1 h-3 w-3 cursor-nwse-resize`} />
        ),
        bottom: (
          <span className={`${handleClass} -bottom-1 left-3 right-3 h-2 cursor-ns-resize`} />
        ),
        bottomLeft: (
          <span className={`${handleClass} -bottom-1 -left-1 h-3 w-3 cursor-nesw-resize`} />
        ),
        left: (
          <span className={`${handleClass} -left-1 bottom-3 top-3 w-2 cursor-ew-resize`} />
        )
      }}
    >
      <div
        className={cn(
          'flex h-full w-full flex-col overflow-hidden border border-retro-border',
          isActive ? 'bg-retro-surface' : 'bg-retro-surface/90'
        )}
        onMouseDown={onFocus}
      >
        <TitleBar
          title={title}
          isActive={isActive}
          onClose={onClose}
          onMinimize={onMinimize}
          variant="windows"
        />
        <div className="os-window-scrollbar min-h-0 flex-1 overflow-x-hidden overflow-y-auto p-4">
          {children}
        </div>
      </div>
    </Rnd>
  );
}

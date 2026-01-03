'use client';

export function MemoryWindow() {
  return (
    <div className="flex h-full flex-col gap-3 text-sm text-retro-text">
      <h2 className="text-base font-semibold">Memory Vault</h2>
      <p className="text-xs text-retro-accent">
        Your archived conversations and session snapshots will appear here.
      </p>
      <div className="flex flex-1 items-center justify-center border border-retro-border bg-retro-title-active text-xs text-retro-accent">
        No memories yet. Start a chat to begin.
      </div>
    </div>
  );
}

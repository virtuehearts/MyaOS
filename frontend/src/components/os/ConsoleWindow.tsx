'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function ConsoleWindow() {
  return (
    <Card className="flex h-full flex-col border border-retro-border bg-retro-surface text-retro-text">
      <CardHeader className="border-b border-retro-border bg-retro-title-active">
        <CardTitle className="text-sm font-semibold text-retro-text">Console</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col gap-3 text-xs text-retro-accent">
        <div className="font-mono text-[11px] text-retro-text">
          MyaOS Console v1.0
        </div>
        <div className="space-y-2 font-mono text-[11px]">
          <div>&gt; boot status --verbose</div>
          <div className="text-retro-text">✔ Kernel online · ✔ Storage mounted</div>
          <div>&gt; system info</div>
          <div className="text-retro-text">
            MyaOS build 95.4 · 32-bit retro session
          </div>
        </div>
        <div className="mt-auto border-t border-retro-border pt-2 font-mono text-[11px]">
          &gt; _
        </div>
      </CardContent>
    </Card>
  );
}

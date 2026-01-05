'use client';

import type { FormEvent } from 'react';
import { useEffect, useMemo, useState } from 'react';
import { ArrowLeft, ArrowRight, Home, RotateCw } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { API_BASE_URL } from '@/lib/api';

const DEFAULT_HOMEPAGE = 'https://example.com/';

type HistoryState = {
  entries: string[];
  index: number;
};

const isAllowedProtocol = (value: string) => {
  const protocol = value.toLowerCase();
  return protocol === 'http:' || protocol === 'https:';
};

const sanitizeUrl = (value: string) => {
  const trimmed = value.trim();
  if (!trimmed) {
    return DEFAULT_HOMEPAGE;
  }

  const withScheme = /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(trimmed)
    ? trimmed
    : `https://${trimmed}`;

  try {
    const parsed = new URL(withScheme);
    if (!isAllowedProtocol(parsed.protocol)) {
      return null;
    }
    return parsed.toString();
  } catch {
    return null;
  }
};

export function BrowserWindow() {
  const [history, setHistory] = useState<HistoryState>({
    entries: [DEFAULT_HOMEPAGE],
    index: 0
  });
  const [address, setAddress] = useState(DEFAULT_HOMEPAGE);
  const [error, setError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  const currentUrl = history.entries[history.index] ?? DEFAULT_HOMEPAGE;
  const canGoBack = history.index > 0;
  const canGoForward = history.index < history.entries.length - 1;

  useEffect(() => {
    setAddress(currentUrl);
  }, [currentUrl]);

  const frameUrl = useMemo(
    () => `${API_BASE_URL}/proxy?url=${encodeURIComponent(currentUrl)}`,
    [currentUrl]
  );

  const navigateTo = (nextUrl: string) => {
    setHistory((prev) => {
      const entries = [...prev.entries.slice(0, prev.index + 1), nextUrl];
      return { entries, index: entries.length - 1 };
    });
    setError(null);
  };

  const handleSubmit = (event?: FormEvent) => {
    event?.preventDefault();
    const sanitized = sanitizeUrl(address);
    if (!sanitized) {
      setError('Enter a valid http(s) URL.');
      return;
    }
    navigateTo(sanitized);
  };

  return (
    <div className="flex h-full flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2 border border-retro-border bg-retro-title-active p-2 text-retro-text">
        <Button
          type="button"
          size="sm"
          variant="ghost"
          onClick={() =>
            setHistory((prev) => ({
              ...prev,
              index: Math.max(prev.index - 1, 0)
            }))
          }
          disabled={!canGoBack}
          aria-label="Back"
        >
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <Button
          type="button"
          size="sm"
          variant="ghost"
          onClick={() =>
            setHistory((prev) => ({
              ...prev,
              index: Math.min(prev.index + 1, prev.entries.length - 1)
            }))
          }
          disabled={!canGoForward}
          aria-label="Forward"
        >
          <ArrowRight className="h-4 w-4" />
        </Button>
        <Button
          type="button"
          size="sm"
          variant="ghost"
          onClick={() => setReloadKey((prev) => prev + 1)}
          aria-label="Refresh"
        >
          <RotateCw className="h-4 w-4" />
        </Button>
        <Button
          type="button"
          size="sm"
          variant="ghost"
          onClick={() => navigateTo(DEFAULT_HOMEPAGE)}
          aria-label="Home"
        >
          <Home className="h-4 w-4" />
        </Button>
        <form className="flex min-w-[200px] flex-1 items-center gap-2" onSubmit={handleSubmit}>
          <Input
            className="h-8 flex-1 bg-retro-surface text-xs"
            value={address}
            onChange={(event) => setAddress(event.target.value)}
            placeholder="Enter a URL"
            aria-label="Browser address"
          />
          <Button type="submit" size="sm" variant="outline">
            Go
          </Button>
        </form>
      </div>
      {error ? (
        <div className="text-xs text-red-200">{error}</div>
      ) : (
        <div className="text-xs text-retro-accent">
          Loading via secure proxy to avoid iframe restrictions.
        </div>
      )}
      <div className="flex min-h-0 flex-1 overflow-hidden border border-retro-border bg-black">
        <iframe
          key={reloadKey}
          title="Browser viewport"
          src={frameUrl}
          className="h-full w-full border-0"
          referrerPolicy="no-referrer"
        />
      </div>
    </div>
  );
}

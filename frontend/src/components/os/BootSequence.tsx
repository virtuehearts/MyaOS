'use client';

import { useEffect, useMemo, useState } from 'react';

import { OsShell } from '@/components/os/OsShell';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { useApiKeyStore } from '@/store/apiKeyStore';
import { useAuthStore } from '@/store/authStore';

const BOOT_DURATION_MS = 2400;

type BootStage = 'boot' | 'apiKey' | 'login' | 'desktop';

export function BootSequence() {
  const { apiKey, setApiKey } = useApiKeyStore();
  const { setUser } = useAuthStore();
  const [stage, setStage] = useState<BootStage>('boot');
  const [bootTick, setBootTick] = useState(0);
  const [keyInput, setKeyInput] = useState(apiKey);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setStage('apiKey');
    }, BOOT_DURATION_MS);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    const interval = window.setInterval(() => {
      setBootTick((tick) => (tick + 1) % 4);
    }, 500);
    return () => window.clearInterval(interval);
  }, []);

  const bootMessage = useMemo(() => {
    const dots = '.'.repeat(bootTick);
    return `Initializing MyaOS${dots}`;
  }, [bootTick]);

  const handleKeySubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmedKey = keyInput.trim();
    if (!trimmedKey) {
      return;
    }
    setApiKey(trimmedKey);
    setStage('login');
  };

  const handleLoginSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmedName = username.trim() || 'Operator';
    setUser({
      user_id: 'local-user',
      email: `${trimmedName.toLowerCase().replace(/\s+/g, '.')}@myaos.local`,
      name: trimmedName,
      created_at: new Date().toISOString()
    });
    setStage('desktop');
  };

  if (stage === 'desktop') {
    return <OsShell />;
  }

  return (
    <div className="flex min-h-[100dvh] items-center justify-center bg-retro-bg p-6 text-retro-text">
      {stage === 'boot' && (
        <div className="flex w-full max-w-xl flex-col gap-6 text-center">
          <p className="text-xs uppercase tracking-[0.4em] text-retro-accent">
            System Boot
          </p>
          <h1 className="text-3xl font-semibold">{bootMessage}</h1>
          <div className="flex items-center justify-center gap-3 text-sm text-retro-text/80">
            <span className="h-2 w-2 animate-pulse rounded-full bg-retro-accent" />
            <span className="h-2 w-2 animate-pulse rounded-full bg-retro-accent [animation-delay:200ms]" />
            <span className="h-2 w-2 animate-pulse rounded-full bg-retro-accent [animation-delay:400ms]" />
          </div>
          <p className="text-sm text-retro-text/70">
            Loading interface modules, memory lattice, and shell services.
          </p>
        </div>
      )}

      {stage === 'apiKey' && (
        <Card className="mya-panel w-full max-w-md space-y-6 border border-retro-border p-6">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-retro-accent">
              Connectivity
            </p>
            <h2 className="mt-2 text-2xl font-semibold">Enter OpenRouter Key</h2>
            <p className="mt-2 text-sm text-retro-text/80">
              Provide your API key to enable cloud model access. Stored locally on
              this device.
            </p>
          </div>
          <form className="space-y-4" onSubmit={handleKeySubmit}>
            <Input
              type="password"
              placeholder="sk-or-..."
              value={keyInput}
              onChange={(event) => setKeyInput(event.target.value)}
              required
            />
            <Button type="submit" className="w-full">
              Save &amp; Continue
            </Button>
          </form>
        </Card>
      )}

      {stage === 'login' && (
        <Card className="mya-panel w-full max-w-md space-y-6 border border-retro-border p-6">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-retro-accent">
              Local Access
            </p>
            <h2 className="mt-2 text-2xl font-semibold">Mock Login</h2>
            <p className="mt-2 text-sm text-retro-text/80">
              Enter any credentials to unlock the desktop environment.
            </p>
          </div>
          <form className="space-y-4" onSubmit={handleLoginSubmit}>
            <Input
              placeholder="Username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              required
            />
            <Input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              required
            />
            <Button type="submit" className="w-full">
              Enter MyaOS
            </Button>
          </form>
        </Card>
      )}
    </div>
  );
}

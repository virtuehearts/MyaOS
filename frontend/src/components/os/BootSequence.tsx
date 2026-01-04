'use client';

import { useEffect, useMemo, useState } from 'react';

import { OsShell } from '@/components/os/OsShell';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { useAuthStore } from '@/store/authStore';

const BOOT_DURATION_MS = 2400;

const DEFAULT_BOOT_USERNAME = process.env.NEXT_PUBLIC_BOOT_USERNAME ?? 'admin';
const DEFAULT_BOOT_PASSWORD = process.env.NEXT_PUBLIC_BOOT_PASSWORD ?? 'password';

type BootStage = 'boot' | 'login' | 'desktop';

export function BootSequence() {
  const { setUser } = useAuthStore();
  const [stage, setStage] = useState<BootStage>('boot');
  const [bootTick, setBootTick] = useState(0);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loginError, setLoginError] = useState('');

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setStage('login');
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
    return `Boot sequence${dots}`;
  }, [bootTick]);

  const handleLoginSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmedName = username.trim();
    const trimmedPassword = password;
    const expectedUsername = DEFAULT_BOOT_USERNAME.trim();
    const expectedPassword = DEFAULT_BOOT_PASSWORD;

    if (
      trimmedName.toLowerCase() !== expectedUsername.toLowerCase() ||
      trimmedPassword !== expectedPassword
    ) {
      setLoginError('Access denied. Verify credentials and try again.');
      return;
    }

    setLoginError('');
    const displayName = trimmedName || 'Operator';
    setUser({
      user_id: 'local-user',
      email: `${displayName.toLowerCase().replace(/\s+/g, '.')}@myaos.local`,
      name: displayName,
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
        <Card className="mya-panel w-full max-w-2xl space-y-6 border border-retro-border p-6 font-mono">
          <div className="space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-3 text-xs uppercase tracking-[0.35em] text-retro-accent">
              <span>MyaOS BIOS</span>
              <span>v3.4.1</span>
            </div>
            <h1 className="text-2xl font-semibold text-retro-text">
              MyaOS System Firmware
            </h1>
            <p className="text-xs text-retro-text/70">
              Copyright (c) 2042 MyaOS Labs. All rights reserved.
            </p>
          </div>
          <div className="space-y-4 text-sm text-retro-text/80">
            <p className="text-retro-text/90">{bootMessage}</p>
            <div className="grid gap-2">
              <div className="flex justify-between">
                <span>CPU:</span>
                <span>NeuralCore X12 @ 4.2GHz</span>
              </div>
              <div className="flex justify-between">
                <span>Memory Test:</span>
                <span>65536MB OK</span>
              </div>
              <div className="flex justify-between">
                <span>System Bus:</span>
                <span>Link Stable</span>
              </div>
            </div>
            <div className="space-y-2 rounded border border-retro-border/70 bg-retro-surface/40 p-3">
              <p className="text-xs uppercase tracking-[0.3em] text-retro-accent">
                Detected Devices
              </p>
              <ul className="space-y-1 text-xs text-retro-text/80">
                <li>[00] Storage Array: MYA-SSD 2TB - OK</li>
                <li>[01] Interface Module: Holo-IO Bridge - OK</li>
                <li>[02] Secure TPM: MYA-TPM 4.0 - OK</li>
                <li>[03] Network Link: Quantum Relay - OK</li>
              </ul>
            </div>
            <div className="flex items-center gap-3 text-xs text-retro-text/70">
              <span className="h-2 w-2 animate-pulse rounded-full bg-retro-accent" />
              <span>Press DEL to enter setup · Auto boot in progress</span>
            </div>
          </div>
        </Card>
      )}

      {stage === 'login' && (
        <Card className="mya-panel w-full max-w-md space-y-6 border border-retro-border p-6 font-mono">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-retro-accent">
              Secure Access
            </p>
            <h2 className="mt-2 text-2xl font-semibold">MYAOS SECURE ACCESS</h2>
            <p className="mt-2 text-sm text-retro-text/80">
              Authorized operator credentials required. Access events are logged.
            </p>
          </div>
          <form className="space-y-4" onSubmit={handleLoginSubmit}>
            <Input
              placeholder="Username"
              value={username}
              onChange={(event) => {
                setUsername(event.target.value);
                setLoginError('');
              }}
              required
            />
            <Input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(event) => {
                setPassword(event.target.value);
                setLoginError('');
              }}
              required
            />
            {loginError ? (
              <p className="text-xs text-retro-accent">{loginError}</p>
            ) : null}
            <Button type="submit" className="w-full">
              Enter MyaOS
            </Button>
          </form>
        </Card>
      )}
    </div>
  );
}

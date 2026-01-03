'use client';

import { useState } from 'react';

import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { apiRequest } from '@/lib/api';
import { useAuthStore } from '@/store/authStore';

interface AuthResponse {
  token: string;
  user: {
    user_id: string;
    email: string;
    name?: string | null;
    created_at: string;
  };
  expires_at: string;
}

interface OAuthStartResponse {
  auth_url: string;
}

export function AuthOverlay() {
  const { setSession } = useAuthStore();
  const [isRegister, setIsRegister] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);
    setIsSubmitting(true);

    try {
      const payload = isRegister
        ? { email, password, name: name || null }
        : { email, password };
      const response = await apiRequest<AuthResponse>(
        isRegister ? '/auth/register' : '/auth/login',
        {
          method: 'POST',
          body: JSON.stringify(payload)
        }
      );
      setSession(response.token, response.user);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Unable to sign in right now.');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleOAuth = async (provider: string) => {
    setError(null);
    setIsSubmitting(true);
    try {
      const response = await apiRequest<OAuthStartResponse>('/auth/oauth/start', {
        method: 'POST',
        body: JSON.stringify({ provider })
      });
      window.location.href = response.auth_url;
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('OAuth is unavailable right now.');
      }
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-retro-bg/90 p-6">
      <Card className="mya-panel w-full max-w-md space-y-6 border border-retro-border p-6">
        <div>
          <p className="text-sm uppercase tracking-[0.25em] text-retro-accent">
            MyaOS Secure Access
          </p>
          <h1 className="mt-2 text-2xl font-semibold">
            {isRegister ? 'Create your session' : 'Welcome back'}
          </h1>
          <p className="mt-2 text-sm text-retro-text/80">
            {isRegister
              ? 'Register with email and keep your memories private.'
              : 'Sign in to access your secure workspace and memory vault.'}
          </p>
        </div>

        <form className="space-y-4" onSubmit={handleSubmit}>
          {isRegister && (
            <Input
              placeholder="Name"
              value={name}
              onChange={(event) => setName(event.target.value)}
              disabled={isSubmitting}
            />
          )}
          <Input
            type="email"
            placeholder="Email address"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            disabled={isSubmitting}
            required
          />
          <Input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            disabled={isSubmitting}
            minLength={8}
            required
          />
          {error && <p className="text-sm text-red-300">{error}</p>}
          <Button type="submit" className="w-full" disabled={isSubmitting}>
            {isRegister ? 'Create account' : 'Sign in'}
          </Button>
        </form>

        <div className="space-y-3">
          <p className="text-xs uppercase tracking-[0.2em] text-retro-accent">
            Or continue with
          </p>
          <div className="grid gap-3 sm:grid-cols-2">
            <Button
              type="button"
              variant="outline"
              onClick={() => handleOAuth('google')}
              disabled={isSubmitting}
            >
              Google OAuth
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={() => handleOAuth('github')}
              disabled={isSubmitting}
            >
              GitHub OAuth
            </Button>
          </div>
        </div>

        <div className="text-sm text-retro-accent">
          {isRegister ? (
            <button
              type="button"
              className="text-retro-text"
              onClick={() => setIsRegister(false)}
              disabled={isSubmitting}
            >
              Already have an account? Sign in
            </button>
          ) : (
            <button
              type="button"
              className="text-retro-text"
              onClick={() => setIsRegister(true)}
              disabled={isSubmitting}
            >
              New here? Create an account
            </button>
          )}
        </div>
      </Card>
    </div>
  );
}

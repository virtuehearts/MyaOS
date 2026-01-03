'use client';

import { useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';

import { apiRequest } from '@/lib/api';
import { AuthUser, useAuthStore } from '@/store/authStore';

export default function OAuthCallbackPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { setSession, clearSession } = useAuthStore();
  const [message, setMessage] = useState('Completing sign-in…');

  useEffect(() => {
    const token = searchParams.get('token');
    if (!token) {
      setMessage('Missing token from OAuth provider.');
      return;
    }

    const finalize = async () => {
      try {
        const response = await apiRequest<{ user: AuthUser }>('/auth/me', {
          headers: { Authorization: `Bearer ${token}` }
        });
        setSession(token, response.user);
        router.replace('/');
      } catch {
        clearSession();
        setMessage('Unable to verify session. Please try again.');
      }
    };

    finalize();
  }, [searchParams, setSession, router, clearSession]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-retro-bg text-sm text-retro-text">
      {message}
    </div>
  );
}

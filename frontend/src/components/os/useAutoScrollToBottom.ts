'use client';

import { useEffect } from 'react';
import type { DependencyList, RefObject } from 'react';

const defaultScrollBehavior: ScrollBehavior = 'smooth';

export function useAutoScrollToBottom(
  ref: RefObject<HTMLElement | null>,
  deps: DependencyList,
  behavior: ScrollBehavior = defaultScrollBehavior
) {
  useEffect(() => {
    ref.current?.scrollIntoView({ behavior });
  }, [behavior, ref, ...deps]);
}

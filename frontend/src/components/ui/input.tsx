import * as React from 'react';

import { cn } from '@/lib/utils';

const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, type, ...props }, ref) => (
    <input
      ref={ref}
      type={type}
      className={cn(
        'flex h-10 w-full border border-retro-border bg-[#0f1012] px-3 py-2 text-sm text-retro-text placeholder:text-retro-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-retro-accent',
        className
      )}
      {...props}
    />
  )
);
Input.displayName = 'Input';

export { Input };

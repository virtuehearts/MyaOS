import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';

import { cn } from '@/lib/utils';

const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 whitespace-nowrap border border-retro-border bg-retro-surface px-2 text-sm font-medium text-retro-text transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-retro-accent disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      variant: {
        default: 'bg-retro-accent text-retro-bg hover:bg-[#13b58a]',
        ghost: 'border-transparent bg-transparent hover:border-retro-border hover:bg-retro-title-active',
        outline:
          'bg-retro-surface text-retro-text hover:bg-retro-title-active hover:text-retro-text'
      },
      size: {
        default: 'h-9 px-3 py-2',
        sm: 'h-8 px-2',
        lg: 'h-10 px-4'
      }
    },
    defaultVariants: {
      variant: 'default',
      size: 'default'
    }
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => (
    <button
      ref={ref}
      className={cn(buttonVariants({ variant, size }), className)}
      {...props}
    />
  )
);

Button.displayName = 'Button';

export { Button, buttonVariants };

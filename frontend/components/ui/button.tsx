import * as React from 'react'
import { Slot } from '@radix-ui/react-slot'
import { cva, type VariantProps } from 'class-variance-authority'

import { cn } from '@/lib/utils'

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-semibold transition-colors disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 shrink-0 [&_svg]:shrink-0 focus-visible:ring-[3px]",
  {
    variants: {
      variant: {
        default: 'bg-[var(--btn-primary)] text-[color:var(--primary-foreground)] hover:bg-[var(--btn-primary-hover)] active:bg-[var(--btn-primary-active)] focus-visible:ring-[color:var(--ring)]',
        destructive:
          'bg-[color:var(--destructive)] text-white hover:bg-[color-mix(in oklab,var(--destructive) 90%,black)] focus-visible:ring-[color:var(--destructive)]',
        outline:
          'border border-[color:var(--accent)] text-[color:var(--accent)] bg-transparent hover:bg-[color-mix(in oklab,var(--accent) 12%,transparent)] focus-visible:ring-[color:var(--ring)]',
        secondary:
          'border border-[color:var(--border)] text-[color:var(--foreground)] bg-transparent hover:bg-[color-mix(in oklab,var(--foreground) 6%,transparent)] focus-visible:ring-[color:var(--ring)]',
        ghost:
          'text-[color:var(--foreground)] hover:bg-[color-mix(in oklab,var(--accent) 10%,transparent)]',
        link: 'text-[#1D4ED8] dark:text-[#93C5FD] underline-offset-4 hover:underline focus-visible:ring-[color:var(--ring)]',
      },
      size: {
        default: 'h-9 px-4 py-2 has-[>svg]:px-3',
        sm: 'h-8 rounded-md gap-1.5 px-3 has-[>svg]:px-2.5',
        lg: 'h-10 rounded-md px-6 has-[>svg]:px-4',
        icon: 'size-9',
        'icon-sm': 'size-8',
        'icon-lg': 'size-10',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  },
)

function Button({
  className,
  variant,
  size,
  asChild = false,
  ...props
}: React.ComponentProps<'button'> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean
  }) {
  const Comp = asChild ? Slot : 'button'

  return (
    <Comp
      data-slot="button"
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
}

export { Button, buttonVariants }

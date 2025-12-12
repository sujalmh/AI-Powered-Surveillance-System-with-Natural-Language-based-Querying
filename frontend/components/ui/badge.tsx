import * as React from 'react'
import { Slot } from '@radix-ui/react-slot'
import { cva, type VariantProps } from 'class-variance-authority'

import { cn } from '@/lib/utils'

const badgeVariants = cva(
  'inline-flex items-center justify-center rounded-md border px-2 py-0.5 text-xs font-medium w-fit whitespace-nowrap shrink-0 [&>svg]:size-3 gap-1 [&>svg]:pointer-events-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive transition-[color,box-shadow] overflow-hidden',
  {
    variants: {
      variant: {
        default:
          'border-transparent bg-[var(--btn-primary)] text-[color:var(--primary-foreground)] [a&]:hover:bg-[var(--btn-primary-hover)]',
        secondary:
          'border border-[color:var(--border)] bg-transparent text-[color:var(--foreground)] [a&]:hover:bg-[color-mix(in oklab,var(--foreground) 6%,transparent)]',
        destructive:
          'border-transparent bg-[color:var(--destructive)] text-white [a&]:hover:bg-[color-mix(in oklab,var(--destructive) 90%,black)] focus-visible:ring-[color:var(--destructive)]',
        outline:
          'border border-[color:var(--accent)] bg-transparent text-[color:var(--accent)] [a&]:hover:bg-[color-mix(in oklab,var(--accent) 12%,transparent)]',
        success:
          'border-transparent bg-[color-mix(in oklab,var(--success) 15%,transparent)] text-[color:var(--success)]',
        warning:
          'border-transparent bg-[color-mix(in oklab,var(--warning) 15%,transparent)] text-[color:var(--warning)]',
        danger:
          'border-transparent bg-[color-mix(in oklab,var(--danger) 15%,transparent)] text-[color:var(--danger)]',
        info:
          'border-transparent bg-[color-mix(in oklab,var(--info) 15%,transparent)] text-[color:var(--info)]',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  },
)

function Badge({
  className,
  variant,
  asChild = false,
  ...props
}: React.ComponentProps<'span'> &
  VariantProps<typeof badgeVariants> & { asChild?: boolean }) {
  const Comp = asChild ? Slot : 'span'

  return (
    <Comp
      data-slot="badge"
      className={cn(badgeVariants({ variant }), className)}
      {...props}
    />
  )
}

export { Badge, badgeVariants }

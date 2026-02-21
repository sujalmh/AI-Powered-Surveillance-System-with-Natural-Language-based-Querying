import * as React from 'react'
import { Slot } from '@radix-ui/react-slot'
import { cva, type VariantProps } from 'class-variance-authority'

import { cn } from '@/lib/utils'

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-semibold transition-colors disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 shrink-0 [&_svg]:shrink-0 focus-visible:ring-[3px]",
  {
    variants: {
      variant: {
        default: 'bg-gradient-to-br from-green-800 to-green-700 text-white shadow-md hover:shadow-lg active:scale-95 transition-all focus-visible:ring-[color:var(--ring)]',
        destructive:
          'bg-[color:var(--destructive)] text-white hover:bg-[color-mix(in oklab,var(--destructive) 90%,black)] shadow-sm focus-visible:ring-[color:var(--destructive)]',
        outline:
          'border border-[color:var(--accent)] text-[color:var(--accent)] bg-transparent hover:bg-[color-mix(in oklab,var(--accent) 12%,transparent)] focus-visible:ring-[color:var(--ring)]',
        secondary:
          'bg-white dark:bg-stone-800 border border-stone-200 dark:border-stone-700 text-stone-900 dark:text-white hover:bg-stone-100 dark:hover:bg-stone-700 shadow-sm focus-visible:ring-[color:var(--ring)] transition-colors',
        ghost:
          'text-stone-600 dark:text-stone-300 hover:bg-stone-100 dark:hover:bg-stone-800 hover:text-stone-900 dark:hover:text-white transition-colors',
        link: 'text-[#1B4332] dark:text-[#86EFAC] underline-offset-4 hover:underline focus-visible:ring-[color:var(--ring)]',
      },
      size: {
        default: 'h-10 px-4 py-2 has-[>svg]:px-3 rounded-lg',
        sm: 'h-8 rounded-md gap-1.5 px-3 has-[>svg]:px-2.5',
        lg: 'h-11 rounded-lg px-6 has-[>svg]:px-4',
        icon: 'size-10 rounded-full shadow-sm bg-white dark:bg-stone-800 border border-stone-200 dark:border-stone-700 hover:bg-stone-50 dark:hover:bg-stone-700 transition-colors',
        'icon-sm': 'size-8 rounded-full',
        'icon-lg': 'size-12 rounded-full',
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

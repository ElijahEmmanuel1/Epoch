declare module '*.module.css' {
  const classes: Record<string, string>
  export default classes
}

declare module 'react-katex' {
  import type { FC } from 'react'
  interface KatexProps {
    math: string
    block?: boolean
    errorColor?: string
    renderError?: (error: Error) => JSX.Element
  }
  export const InlineMath: FC<KatexProps>
  export const BlockMath: FC<KatexProps>
}

import { cn } from "@/lib/utils";
import React from "react";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: "primary" | "secondary" | "outline" | "ghost";
    size?: "sm" | "md" | "lg";
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
    ({ className, variant = "primary", size = "md", ...props }, ref) => {
        return (
            <button
                ref={ref}
                className={cn(
                    "relative inline-flex items-center justify-center rounded-lg font-medium transition-all focus:outline-none focus:ring-2 focus:ring-zinc-400 focus:ring-offset-2 focus:ring-offset-zinc-900 disabled:opacity-50 disabled:pointer-events-none",

                    // Variants
                    variant === "primary" &&
                    "bg-gradient-to-b from-zinc-50 to-zinc-300 text-zinc-900 shadow-[0px_0px_1px_1px_rgba(255,255,255,0.8)_inset,0px_1px_2px_0px_rgba(255,255,255,0.5)_inset] hover:brightness-105 active:scale-[0.98]",

                    variant === "secondary" &&
                    "bg-white/5 border border-white/10 text-zinc-200 hover:bg-white/10 hover:border-white/20 backdrop-blur-sm",

                    variant === "outline" &&
                    "border border-zinc-700 bg-transparent hover:bg-zinc-800 text-zinc-100",

                    variant === "ghost" &&
                    "bg-transparent hover:bg-zinc-800/50 text-zinc-300 hover:text-white",

                    // Sizes
                    size === "sm" && "h-8 px-3 text-xs",
                    size === "md" && "h-10 px-4 text-sm",
                    size === "lg" && "h-12 px-6 text-base",

                    className
                )}
                {...props}
            />
        );
    }
);
Button.displayName = "Button";

import { cn } from "@/lib/utils";
import React from "react";

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
    variant?: "default" | "success" | "warning" | "error" | "info" | "outline";
    size?: "sm" | "md";
}

export function Badge({ className, variant = "default", size = "sm", children, ...props }: BadgeProps) {
    return (
        <span
            className={cn(
                "inline-flex items-center font-medium rounded-full whitespace-nowrap",
                size === "sm" && "px-2 py-0.5 text-[10px]",
                size === "md" && "px-2.5 py-1 text-xs",

                variant === "default" && "bg-muted text-muted-foreground",
                variant === "success" && "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20",
                variant === "warning" && "bg-amber-500/10 text-amber-400 border border-amber-500/20",
                variant === "error" && "bg-red-500/10 text-red-400 border border-red-500/20",
                variant === "info" && "bg-blue-500/10 text-blue-400 border border-blue-500/20",
                variant === "outline" && "border border-border text-muted-foreground",
                className
            )}
            {...props}
        >
            {children}
        </span>
    );
}

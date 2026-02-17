import { cn } from "@/lib/utils";
import React from "react";

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
    variant?: "default" | "glass" | "bordered";
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
    ({ className, variant = "default", ...props }, ref) => {
        return (
            <div
                ref={ref}
                className={cn(
                    "rounded-xl transition-all",
                    variant === "default" && "bg-card text-card-foreground border border-border shadow-sm",
                    variant === "glass" && "glass text-zinc-100",
                    variant === "bordered" && "border border-zinc-800 bg-zinc-900/50",
                    className
                )}
                {...props}
            />
        );
    }
);
Card.displayName = "Card";

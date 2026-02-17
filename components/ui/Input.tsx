import { cn } from "@/lib/utils";
import React from "react";

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
    label?: string;
    error?: string;
    icon?: React.ReactNode;
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
    ({ className, label, error, icon, id, ...props }, ref) => {
        const inputId = id || label?.toLowerCase().replace(/\s+/g, "-");
        return (
            <div className="space-y-1.5">
                {label && (
                    <label
                        htmlFor={inputId}
                        className="block text-sm font-medium text-foreground"
                    >
                        {label}
                    </label>
                )}
                <div className="relative">
                    {icon && (
                        <div className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">
                            {icon}
                        </div>
                    )}
                    <input
                        ref={ref}
                        id={inputId}
                        className={cn(
                            "flex h-10 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm text-foreground",
                            "placeholder:text-muted-foreground",
                            "focus:outline-none focus:ring-2 focus:ring-zinc-400 focus:ring-offset-2 focus:ring-offset-background",
                            "disabled:cursor-not-allowed disabled:opacity-50",
                            "transition-colors",
                            icon && "pl-10",
                            error && "border-red-500 focus:ring-red-500",
                            className
                        )}
                        aria-invalid={error ? "true" : undefined}
                        aria-describedby={error ? `${inputId}-error` : undefined}
                        {...props}
                    />
                </div>
                {error && (
                    <p id={`${inputId}-error`} className="text-xs text-red-400" role="alert">
                        {error}
                    </p>
                )}
            </div>
        );
    }
);
Input.displayName = "Input";

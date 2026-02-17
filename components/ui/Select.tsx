"use client";

import { cn } from "@/lib/utils";
import React, { useState, useRef, useEffect } from "react";
import { ChevronDown } from "lucide-react";

interface SelectOption {
    value: string;
    label: string;
    description?: string;
}

interface SelectProps {
    options: SelectOption[];
    value?: string;
    onChange?: (value: string) => void;
    placeholder?: string;
    label?: string;
    className?: string;
}

export function Select({ options, value, onChange, placeholder = "Select...", label, className }: SelectProps) {
    const [open, setOpen] = useState(false);
    const ref = useRef<HTMLDivElement>(null);
    const selected = options.find((o) => o.value === value);

    useEffect(() => {
        const handleClick = (e: MouseEvent) => {
            if (ref.current && !ref.current.contains(e.target as Node)) {
                setOpen(false);
            }
        };
        document.addEventListener("mousedown", handleClick);
        return () => document.removeEventListener("mousedown", handleClick);
    }, []);

    return (
        <div className={cn("space-y-1.5", className)} ref={ref}>
            {label && (
                <label className="block text-sm font-medium text-foreground">{label}</label>
            )}
            <button
                type="button"
                onClick={() => setOpen(!open)}
                className={cn(
                    "flex h-10 w-full items-center justify-between rounded-lg border border-border bg-card px-3 py-2 text-sm",
                    "focus:outline-none focus:ring-2 focus:ring-zinc-400 focus:ring-offset-2 focus:ring-offset-background",
                    "transition-colors",
                    selected ? "text-foreground" : "text-muted-foreground"
                )}
                aria-expanded={open}
                aria-haspopup="listbox"
            >
                <span className="truncate">{selected?.label || placeholder}</span>
                <ChevronDown className={cn("h-4 w-4 text-muted-foreground transition-transform", open && "rotate-180")} />
            </button>
            {open && (
                <div
                    className="absolute z-50 mt-1 w-full min-w-[8rem] rounded-xl border border-border bg-card shadow-xl animate-in fade-in-0 zoom-in-95 duration-150"
                    role="listbox"
                >
                    <div className="p-1">
                        {options.map((option) => (
                            <button
                                key={option.value}
                                onClick={() => {
                                    onChange?.(option.value);
                                    setOpen(false);
                                }}
                                className={cn(
                                    "flex w-full items-start gap-2 rounded-lg px-3 py-2 text-sm text-left transition-colors",
                                    option.value === value
                                        ? "bg-muted text-foreground"
                                        : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
                                )}
                                role="option"
                                aria-selected={option.value === value}
                            >
                                <div>
                                    <div className="font-medium">{option.label}</div>
                                    {option.description && (
                                        <div className="text-xs text-muted-foreground mt-0.5">{option.description}</div>
                                    )}
                                </div>
                            </button>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

"use client";

import { cn } from "@/lib/utils";
import React, { useState, useRef, useEffect } from "react";

interface TooltipProps {
    content: string;
    children: React.ReactNode;
    side?: "top" | "bottom" | "left" | "right";
    className?: string;
}

export function Tooltip({ content, children, side = "top", className }: TooltipProps) {
    const [visible, setVisible] = useState(false);
    const timeoutRef = useRef<ReturnType<typeof setTimeout>>(undefined);

    const show = () => {
        timeoutRef.current = setTimeout(() => setVisible(true), 300);
    };

    const hide = () => {
        clearTimeout(timeoutRef.current);
        setVisible(false);
    };

    useEffect(() => {
        return () => clearTimeout(timeoutRef.current);
    }, []);

    return (
        <div className="relative inline-flex" onMouseEnter={show} onMouseLeave={hide} onFocus={show} onBlur={hide}>
            {children}
            {visible && (
                <div
                    role="tooltip"
                    className={cn(
                        "absolute z-50 px-2.5 py-1.5 text-xs font-medium text-foreground bg-card border border-border rounded-lg shadow-lg whitespace-nowrap",
                        "animate-in fade-in-0 zoom-in-95 duration-150",
                        side === "top" && "bottom-full left-1/2 -translate-x-1/2 mb-2",
                        side === "bottom" && "top-full left-1/2 -translate-x-1/2 mt-2",
                        side === "left" && "right-full top-1/2 -translate-y-1/2 mr-2",
                        side === "right" && "left-full top-1/2 -translate-y-1/2 ml-2",
                        className
                    )}
                >
                    {content}
                </div>
            )}
        </div>
    );
}

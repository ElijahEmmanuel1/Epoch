"use client";

import { cn } from "@/lib/utils";
import { X } from "lucide-react";
import React, { useEffect, useCallback } from "react";

interface ModalProps {
    open: boolean;
    onClose: () => void;
    children: React.ReactNode;
    className?: string;
}

export function Modal({ open, onClose, children, className }: ModalProps) {
    const handleEscape = useCallback(
        (e: KeyboardEvent) => {
            if (e.key === "Escape") onClose();
        },
        [onClose]
    );

    useEffect(() => {
        if (open) {
            document.addEventListener("keydown", handleEscape);
            document.body.style.overflow = "hidden";
        }
        return () => {
            document.removeEventListener("keydown", handleEscape);
            document.body.style.overflow = "";
        };
    }, [open, handleEscape]);

    if (!open) return null;

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center">
            {/* Backdrop */}
            <div
                className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-in fade-in-0 duration-200"
                onClick={onClose}
                aria-hidden="true"
            />
            {/* Content */}
            <div
                className={cn(
                    "relative z-10 w-full max-w-lg mx-4 rounded-2xl border border-border bg-card p-6 shadow-2xl",
                    "animate-in fade-in-0 zoom-in-95 duration-200",
                    className
                )}
                role="dialog"
                aria-modal="true"
            >
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 p-1 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
                    aria-label="Close dialog"
                >
                    <X className="h-4 w-4" />
                </button>
                {children}
            </div>
        </div>
    );
}

export function ModalHeader({ children, className }: { children: React.ReactNode; className?: string }) {
    return <div className={cn("mb-4 space-y-1.5", className)}>{children}</div>;
}

export function ModalTitle({ children, className }: { children: React.ReactNode; className?: string }) {
    return <h2 className={cn("text-lg font-semibold text-foreground", className)}>{children}</h2>;
}

export function ModalDescription({ children, className }: { children: React.ReactNode; className?: string }) {
    return <p className={cn("text-sm text-muted-foreground", className)}>{children}</p>;
}

export function ModalFooter({ children, className }: { children: React.ReactNode; className?: string }) {
    return <div className={cn("mt-6 flex items-center justify-end gap-3", className)}>{children}</div>;
}

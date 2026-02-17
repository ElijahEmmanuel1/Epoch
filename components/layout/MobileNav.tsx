"use client";

import { cn } from "@/lib/utils";
import { Menu, X } from "lucide-react";
import { useState, useEffect } from "react";
import { usePathname } from "next/navigation";

interface MobileNavProps {
    children: React.ReactNode;
}

export function MobileNav({ children }: MobileNavProps) {
    const [open, setOpen] = useState(false);
    const pathname = usePathname();

    // Close on route change
    useEffect(() => {
        setOpen(false);
    }, [pathname]);

    // Lock body scroll when open
    useEffect(() => {
        if (open) {
            document.body.style.overflow = "hidden";
        } else {
            document.body.style.overflow = "";
        }
        return () => {
            document.body.style.overflow = "";
        };
    }, [open]);

    return (
        <>
            {/* Mobile trigger */}
            <button
                onClick={() => setOpen(true)}
                className="md:hidden fixed top-4 left-4 z-50 p-2 rounded-xl bg-card border border-border shadow-lg text-foreground hover:bg-muted transition-colors"
                aria-label="Open navigation"
            >
                <Menu className="h-5 w-5" />
            </button>

            {/* Overlay */}
            {open && (
                <div
                    className="md:hidden fixed inset-0 z-[90] bg-black/60 backdrop-blur-sm animate-in fade-in-0 duration-200"
                    onClick={() => setOpen(false)}
                    aria-hidden="true"
                />
            )}

            {/* Drawer */}
            <div
                className={cn(
                    "md:hidden fixed inset-y-0 left-0 z-[95] w-72 bg-background border-r border-border",
                    "transition-transform duration-300 ease-out",
                    open ? "translate-x-0" : "-translate-x-full"
                )}
            >
                <button
                    onClick={() => setOpen(false)}
                    className="absolute top-4 right-4 p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
                    aria-label="Close navigation"
                >
                    <X className="h-5 w-5" />
                </button>
                {children}
            </div>
        </>
    );
}

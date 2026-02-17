"use client";

import { cn } from "@/lib/utils";
import React, { createContext, useContext, useState } from "react";

interface TabsContextType {
    activeTab: string;
    setActiveTab: (id: string) => void;
}

const TabsContext = createContext<TabsContextType | null>(null);

function useTabs() {
    const ctx = useContext(TabsContext);
    if (!ctx) throw new Error("Tabs components must be used within <Tabs>");
    return ctx;
}

interface TabsProps {
    defaultValue: string;
    children: React.ReactNode;
    className?: string;
}

export function Tabs({ defaultValue, children, className }: TabsProps) {
    const [activeTab, setActiveTab] = useState(defaultValue);
    return (
        <TabsContext.Provider value={{ activeTab, setActiveTab }}>
            <div className={cn("space-y-4", className)}>{children}</div>
        </TabsContext.Provider>
    );
}

export function TabsList({ children, className }: { children: React.ReactNode; className?: string }) {
    return (
        <div
            className={cn(
                "inline-flex items-center gap-1 rounded-lg bg-muted/50 p-1 border border-border",
                className
            )}
            role="tablist"
        >
            {children}
        </div>
    );
}

interface TabTriggerProps {
    value: string;
    children: React.ReactNode;
    className?: string;
}

export function TabTrigger({ value, children, className }: TabTriggerProps) {
    const { activeTab, setActiveTab } = useTabs();
    const isActive = activeTab === value;

    return (
        <button
            role="tab"
            aria-selected={isActive}
            onClick={() => setActiveTab(value)}
            className={cn(
                "inline-flex items-center justify-center rounded-md px-3 py-1.5 text-sm font-medium transition-all",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:ring-offset-2 focus-visible:ring-offset-background",
                isActive
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground hover:bg-background/50",
                className
            )}
        >
            {children}
        </button>
    );
}

interface TabContentProps {
    value: string;
    children: React.ReactNode;
    className?: string;
}

export function TabContent({ value, children, className }: TabContentProps) {
    const { activeTab } = useTabs();
    if (activeTab !== value) return null;

    return (
        <div
            role="tabpanel"
            className={cn("animate-in fade-in-50 duration-200", className)}
        >
            {children}
        </div>
    );
}

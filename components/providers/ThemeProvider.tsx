"use client";

import React, { createContext, useContext, useEffect, useState } from "react";

type Theme = "dark" | "light" | "system";

interface ThemeContextType {
    theme: Theme;
    setTheme: (theme: Theme) => void;
    resolvedTheme: "dark" | "light";
}

const ThemeContext = createContext<ThemeContextType>({
    theme: "dark",
    setTheme: () => { },
    resolvedTheme: "dark",
});

export function useTheme() {
    return useContext(ThemeContext);
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
    const [theme, setTheme] = useState<Theme>("dark");
    const [resolvedTheme, setResolvedTheme] = useState<"dark" | "light">("dark");

    useEffect(() => {
        const stored = localStorage.getItem("epoch-theme") as Theme | null;
        if (stored) setTheme(stored);
    }, []);

    useEffect(() => {
        localStorage.setItem("epoch-theme", theme);

        const root = document.documentElement;
        if (theme === "system") {
            const mq = window.matchMedia("(prefers-color-scheme: dark)");
            const resolved = mq.matches ? "dark" : "light";
            setResolvedTheme(resolved);
            root.setAttribute("data-theme", resolved);

            const handler = (e: MediaQueryListEvent) => {
                const r = e.matches ? "dark" : "light";
                setResolvedTheme(r);
                root.setAttribute("data-theme", r);
            };
            mq.addEventListener("change", handler);
            return () => mq.removeEventListener("change", handler);
        } else {
            setResolvedTheme(theme);
            root.setAttribute("data-theme", theme);
        }
    }, [theme]);

    return (
        <ThemeContext.Provider value={{ theme, setTheme, resolvedTheme }}>
            {children}
        </ThemeContext.Provider>
    );
}

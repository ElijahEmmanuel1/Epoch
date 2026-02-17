"use client";

import katex from "katex";
import "katex/dist/katex.min.css";
import { useEffect, useRef } from "react";

interface MathProps {
    latex: string;
    block?: boolean;
    className?: string;
}

export function Math({ latex, block = false, className }: MathProps) {
    const ref = useRef<HTMLSpanElement>(null);

    useEffect(() => {
        if (ref.current) {
            katex.render(latex, ref.current, {
                displayMode: block,
                throwOnError: false,
            });
        }
    }, [latex, block]);

    return <span ref={ref} className={className} />;
}

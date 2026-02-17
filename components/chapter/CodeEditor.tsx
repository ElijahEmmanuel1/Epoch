"use client";

import React, { useState, useRef, useCallback } from "react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Play, Copy, Check, RotateCcw, Terminal, ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "@/lib/utils";

interface CodeEditorProps {
    /** Title shown above the editor */
    title?: string;
    /** Language label */
    language?: string;
    /** The starter code the user sees */
    initialCode: string;
    /** Expected output or hint (shown after running) */
    expectedOutput?: string;
    /** Description of the exercise */
    description?: string;
    /** Optional solution code (collapsible) */
    solution?: string;
    /** Hints for the user */
    hints?: string[];
}

export function CodeEditor({
    title = "Code Exercise",
    language = "python",
    initialCode,
    expectedOutput,
    description,
    solution,
    hints,
}: CodeEditorProps) {
    const [code, setCode] = useState(initialCode);
    const [output, setOutput] = useState<string | null>(null);
    const [isRunning, setIsRunning] = useState(false);
    const [copied, setCopied] = useState(false);
    const [showSolution, setShowSolution] = useState(false);
    const [showHints, setShowHints] = useState(false);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    const handleRun = useCallback(() => {
        setIsRunning(true);
        setOutput(null);

        // Simulate execution with a short delay
        // In production: send to a backend API / Pyodide / WebContainer
        setTimeout(() => {
            try {
                const result = simulatePythonExecution(code);
                setOutput(result);
            } catch (err) {
                setOutput(`Error: ${err instanceof Error ? err.message : "Unknown error"}`);
            }
            setIsRunning(false);
        }, 600);
    }, [code]);

    const handleCopy = useCallback(() => {
        navigator.clipboard.writeText(code);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    }, [code]);

    const handleReset = useCallback(() => {
        setCode(initialCode);
        setOutput(null);
    }, [initialCode]);

    const handleKeyDown = useCallback(
        (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
            // Ctrl/Cmd + Enter to run
            if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                e.preventDefault();
                handleRun();
                return;
            }

            // Tab indentation
            if (e.key === "Tab") {
                e.preventDefault();
                const textarea = textareaRef.current;
                if (!textarea) return;
                const start = textarea.selectionStart;
                const end = textarea.selectionEnd;
                const newCode = code.substring(0, start) + "    " + code.substring(end);
                setCode(newCode);
                requestAnimationFrame(() => {
                    textarea.selectionStart = textarea.selectionEnd = start + 4;
                });
            }
        },
        [code, handleRun]
    );

    const lineCount = code.split("\n").length;

    return (
        <Card variant="bordered" className="my-8 overflow-hidden bg-[#0d1117] border-border">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 bg-[#161b22] border-b border-border">
                <div className="flex items-center gap-3">
                    <Terminal className="h-4 w-4 text-emerald-400" />
                    <span className="text-sm font-semibold text-foreground">{title}</span>
                    <Badge variant="outline" size="sm">{language}</Badge>
                </div>
                <div className="flex items-center gap-1.5">
                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={handleCopy}
                        className="h-7 px-2 text-muted-foreground hover:text-foreground"
                    >
                        {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                    </Button>
                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={handleReset}
                        className="h-7 px-2 text-muted-foreground hover:text-foreground"
                    >
                        <RotateCcw className="h-3.5 w-3.5" />
                    </Button>
                    <Button
                        variant="primary"
                        size="sm"
                        onClick={handleRun}
                        disabled={isRunning}
                        className="h-7 px-3 gap-1.5"
                    >
                        <Play className={cn("h-3.5 w-3.5", isRunning && "animate-pulse")} />
                        {isRunning ? "Running..." : "Run"}
                    </Button>
                </div>
            </div>

            {/* Description */}
            {description && (
                <div className="px-4 py-3 border-b border-border bg-blue-500/5">
                    <p className="text-sm text-muted-foreground font-serif leading-relaxed">
                        <span className="text-blue-400 font-sans font-medium mr-1.5">Exercise:</span>
                        {description}
                    </p>
                </div>
            )}

            {/* Editor */}
            <div className="relative flex">
                {/* Line numbers */}
                <div className="flex flex-col items-end py-4 px-3 bg-[#0d1117] border-r border-border select-none min-w-[3rem]" aria-hidden="true">
                    {Array.from({ length: lineCount }, (_, i) => (
                        <span key={i} className="text-[11px] font-mono text-muted-foreground/30 leading-[1.6rem]">
                            {i + 1}
                        </span>
                    ))}
                </div>

                {/* Code textarea */}
                <textarea
                    ref={textareaRef}
                    value={code}
                    onChange={(e) => setCode(e.target.value)}
                    onKeyDown={handleKeyDown}
                    spellCheck={false}
                    autoComplete="off"
                    autoCorrect="off"
                    autoCapitalize="off"
                    className={cn(
                        "flex-1 bg-transparent text-emerald-300 font-mono text-sm leading-[1.6rem] p-4 resize-none",
                        "focus:outline-none caret-emerald-400",
                        "placeholder:text-muted-foreground/30"
                    )}
                    style={{
                        minHeight: `${Math.max(lineCount * 1.6, 6 * 1.6)}rem`,
                        tabSize: 4,
                    }}
                    placeholder="# Write your code here..."
                    aria-label="Code editor"
                />
            </div>

            {/* Keyboard shortcut hint */}
            <div className="px-4 py-1.5 border-t border-border bg-[#161b22] flex items-center justify-between">
                <span className="text-[10px] text-muted-foreground/50 font-mono">
                    ⌘+Enter to run · Tab to indent
                </span>
                {expectedOutput && (
                    <span className="text-[10px] text-muted-foreground/50 font-mono">
                        Expected: {expectedOutput.length > 40 ? expectedOutput.slice(0, 40) + "…" : expectedOutput}
                    </span>
                )}
            </div>

            {/* Output */}
            {output !== null && (
                <div className="border-t border-border">
                    <div className="px-4 py-2 bg-[#161b22] flex items-center gap-2">
                        <Terminal className="h-3.5 w-3.5 text-muted-foreground" />
                        <span className="text-xs font-medium text-muted-foreground">Output</span>
                    </div>
                    <pre className="px-4 py-3 text-sm font-mono text-foreground bg-[#0d1117] whitespace-pre-wrap overflow-x-auto max-h-60">
                        {output}
                    </pre>
                    {expectedOutput && (
                        <div className={cn(
                            "px-4 py-2 border-t border-border text-xs font-mono",
                            output.trim() === expectedOutput.trim()
                                ? "bg-emerald-500/5 text-emerald-400"
                                : "bg-amber-500/5 text-amber-400"
                        )}>
                            {output.trim() === expectedOutput.trim()
                                ? "✓ Output matches expected result!"
                                : "⚠ Output differs from expected. Keep trying!"}
                        </div>
                    )}
                </div>
            )}

            {/* Hints */}
            {hints && hints.length > 0 && (
                <div className="border-t border-border">
                    <button
                        onClick={() => setShowHints(!showHints)}
                        className="flex items-center gap-2 w-full px-4 py-2.5 text-xs font-medium text-amber-400 hover:bg-amber-500/5 transition-colors"
                    >
                        {showHints ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                        {showHints ? "Hide Hints" : `Show Hints (${hints.length})`}
                    </button>
                    {showHints && (
                        <div className="px-4 pb-3 space-y-1.5">
                            {hints.map((hint, i) => (
                                <p key={i} className="text-xs text-muted-foreground font-serif pl-4 border-l-2 border-amber-500/20">
                                    <span className="text-amber-400 font-mono mr-1">Hint {i + 1}:</span>
                                    {hint}
                                </p>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* Solution */}
            {solution && (
                <div className="border-t border-border">
                    <button
                        onClick={() => setShowSolution(!showSolution)}
                        className="flex items-center gap-2 w-full px-4 py-2.5 text-xs font-medium text-violet-400 hover:bg-violet-500/5 transition-colors"
                    >
                        {showSolution ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                        {showSolution ? "Hide Solution" : "Show Solution"}
                    </button>
                    {showSolution && (
                        <pre className="px-4 pb-4 text-sm font-mono text-violet-300/80 bg-violet-500/5 whitespace-pre-wrap overflow-x-auto">
                            {solution}
                        </pre>
                    )}
                </div>
            )}
        </Card>
    );
}

/**
 * Simple Python-like expression simulator.
 * Handles basic operations for educational purposes.
 * In production, replace with Pyodide or a backend API.
 */
function simulatePythonExecution(code: string): string {
    const lines = code.split("\n").filter(l => l.trim() && !l.trim().startsWith("#"));
    const outputs: string[] = [];
    const env: Record<string, number | number[] | string> = {};

    for (const line of lines) {
        const trimmed = line.trim();

        // Handle print statements
        const printMatch = trimmed.match(/^print\s*\((.+)\)$/);
        if (printMatch) {
            const arg = printMatch[1].trim();
            const result = evaluateExpression(arg, env);
            outputs.push(String(result));
            continue;
        }

        // Handle variable assignments: x = expression
        const assignMatch = trimmed.match(/^(\w+)\s*=\s*(.+)$/);
        if (assignMatch) {
            const [, varName, expr] = assignMatch;
            env[varName] = evaluateExpression(expr.trim(), env);
            continue;
        }

        // Handle import (just skip gracefully)
        if (trimmed.startsWith("import ") || trimmed.startsWith("from ")) {
            continue;
        }

        // Handle for loops (basic single-line or detect pattern)
        if (trimmed.startsWith("for ") || trimmed.startsWith("if ") || trimmed.startsWith("def ")) {
            outputs.push(`[Simulation] Complex constructs like '${trimmed.split(" ")[0]}' require a full Python runtime.`);
            outputs.push(`→ Connect to Pyodide or a backend for full execution.`);
            break;
        }
    }

    if (outputs.length === 0) {
        return "(No output — add print() statements to see results)";
    }

    return outputs.join("\n");
}

function evaluateExpression(expr: string, env: Record<string, number | number[] | string>): number | string {
    // Handle string literals
    if ((expr.startsWith('"') && expr.endsWith('"')) || (expr.startsWith("'") && expr.endsWith("'"))) {
        return expr.slice(1, -1);
    }

    // Handle f-strings (basic)
    const fstringMatch = expr.match(/^f["'](.+)["']$/);
    if (fstringMatch) {
        return fstringMatch[1].replace(/\{([^}]+)\}/g, (_, key) => {
            const val = env[key.trim()];
            return val !== undefined ? String(val) : `{${key}}`;
        });
    }

    // Handle max/min
    const maxMatch = expr.match(/^max\((.+)\)$/);
    if (maxMatch) {
        const args = maxMatch[1].split(",").map(a => Number(evaluateExpression(a.trim(), env)));
        return Math.max(...args);
    }

    const minMatch = expr.match(/^min\((.+)\)$/);
    if (minMatch) {
        const args = minMatch[1].split(",").map(a => Number(evaluateExpression(a.trim(), env)));
        return Math.min(...args);
    }

    // Handle round
    const roundMatch = expr.match(/^round\((.+)\)$/);
    if (roundMatch) {
        const args = roundMatch[1].split(",").map(a => evaluateExpression(a.trim(), env));
        return Math.round(Number(args[0]) * Math.pow(10, Number(args[1] || 0))) / Math.pow(10, Number(args[1] || 0));
    }

    // Handle abs
    const absMatch = expr.match(/^abs\((.+)\)$/);
    if (absMatch) {
        return Math.abs(Number(evaluateExpression(absMatch[1].trim(), env)));
    }

    // Handle sum([...])
    const sumMatch = expr.match(/^sum\(\[(.+)\]\)$/);
    if (sumMatch) {
        const vals = sumMatch[1].split(",").map(a => Number(evaluateExpression(a.trim(), env)));
        return vals.reduce((s, v) => s + v, 0);
    }

    // Handle len
    const lenMatch = expr.match(/^len\((.+)\)$/);
    if (lenMatch) {
        const inner = env[lenMatch[1].trim()];
        if (Array.isArray(inner)) return inner.length;
        if (typeof inner === "string") return inner.length;
        return 0;
    }

    // Handle list literals [a, b, c]
    if (expr.startsWith("[") && expr.endsWith("]")) {
        const inner = expr.slice(1, -1);
        const items = inner.split(",").map(a => Number(evaluateExpression(a.trim(), env)));
        return `[${items.join(", ")}]`;
    }

    // Handle basic arithmetic with variables
    try {
        // Replace variable names with their values
        let evalExpr = expr;
        const varNames = Object.keys(env).sort((a, b) => b.length - a.length);
        for (const v of varNames) {
            const val = env[v];
            if (typeof val === "number") {
                evalExpr = evalExpr.replace(new RegExp(`\\b${v}\\b`, "g"), String(val));
            }
        }
        // Replace ** with Math.pow
        evalExpr = evalExpr.replace(/(\d+(?:\.\d+)?)\s*\*\*\s*(\d+(?:\.\d+)?)/g, "Math.pow($1,$2)");

        // Safe eval for simple math
        // eslint-disable-next-line no-new-func
        const result = new Function(`"use strict"; return (${evalExpr});`)();
        if (typeof result === "number") {
            return Number.isInteger(result) ? result : Math.round(result * 10000) / 10000;
        }
        return String(result);
    } catch {
        // Check if it's just a variable reference
        if (expr in env) return env[expr] as number;
        return `<undefined: ${expr}>`;
    }
}

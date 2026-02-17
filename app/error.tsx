"use client";

import { Button } from "@/components/ui/Button";
import { AlertTriangle, RotateCcw } from "lucide-react";

export default function Error({
    error,
    reset,
}: {
    error: Error & { digest?: string };
    reset: () => void;
}) {
    return (
        <div className="min-h-screen flex items-center justify-center bg-background px-6">
            <div className="text-center max-w-md mx-auto space-y-6">
                <div className="inline-flex items-center justify-center h-16 w-16 rounded-2xl bg-red-500/10 border border-red-500/20 mx-auto">
                    <AlertTriangle className="h-7 w-7 text-red-400" />
                </div>

                <div className="space-y-2">
                    <h1 className="text-2xl font-bold tracking-tight text-foreground">Something went wrong</h1>
                    <p className="text-muted-foreground font-serif">
                        An unexpected error occurred. Please try again.
                    </p>
                    {error.digest && (
                        <p className="text-xs text-muted-foreground/50 font-mono">
                            Error ID: {error.digest}
                        </p>
                    )}
                </div>

                <Button onClick={reset} variant="primary" size="md">
                    <RotateCcw className="h-4 w-4 mr-2" />
                    Try Again
                </Button>
            </div>
        </div>
    );
}

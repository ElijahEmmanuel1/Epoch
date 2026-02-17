import { cn } from "@/lib/utils";

interface ProgressProps {
    value: number;
    max?: number;
    size?: "sm" | "md" | "lg";
    variant?: "default" | "success" | "gradient";
    showLabel?: boolean;
    className?: string;
}

export function Progress({
    value,
    max = 100,
    size = "md",
    variant = "default",
    showLabel = false,
    className,
}: ProgressProps) {
    const percentage = Math.min(Math.max((value / max) * 100, 0), 100);

    return (
        <div className={cn("flex items-center gap-3", className)}>
            <div
                className={cn(
                    "relative flex-1 overflow-hidden rounded-full bg-muted",
                    size === "sm" && "h-1",
                    size === "md" && "h-2",
                    size === "lg" && "h-3",
                )}
                role="progressbar"
                aria-valuenow={value}
                aria-valuemin={0}
                aria-valuemax={max}
            >
                <div
                    className={cn(
                        "h-full rounded-full transition-all duration-500 ease-out",
                        variant === "default" && "bg-foreground/80",
                        variant === "success" && "bg-emerald-500",
                        variant === "gradient" && "bg-gradient-to-r from-blue-500 to-violet-500",
                    )}
                    style={{ width: `${percentage}%` }}
                />
            </div>
            {showLabel && (
                <span className="text-xs font-medium text-muted-foreground tabular-nums min-w-[3ch]">
                    {Math.round(percentage)}%
                </span>
            )}
        </div>
    );
}

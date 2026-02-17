import { cn } from "@/lib/utils";

interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
    variant?: "text" | "circular" | "rectangular";
}

export function Skeleton({ className, variant = "text", ...props }: SkeletonProps) {
    return (
        <div
            className={cn(
                "animate-pulse bg-muted",
                variant === "text" && "h-4 rounded",
                variant === "circular" && "rounded-full",
                variant === "rectangular" && "rounded-xl",
                className
            )}
            aria-hidden="true"
            {...props}
        />
    );
}

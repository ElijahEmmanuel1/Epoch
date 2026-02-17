import { cn } from "@/lib/utils";
import React from "react";

interface AvatarProps extends React.HTMLAttributes<HTMLDivElement> {
    src?: string;
    alt?: string;
    fallback: string;
    size?: "sm" | "md" | "lg";
    status?: "online" | "offline" | "busy";
}

export function Avatar({ src, alt, fallback, size = "md", status, className, ...props }: AvatarProps) {
    const [imgError, setImgError] = React.useState(false);

    return (
        <div className={cn("relative inline-flex shrink-0", className)} {...props}>
            <div
                className={cn(
                    "rounded-full flex items-center justify-center font-semibold overflow-hidden",
                    "bg-gradient-to-br from-indigo-500/20 to-violet-500/20 border border-indigo-500/30 text-indigo-300",
                    size === "sm" && "h-7 w-7 text-[10px]",
                    size === "md" && "h-9 w-9 text-xs",
                    size === "lg" && "h-12 w-12 text-sm",
                )}
            >
                {src && !imgError ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                        src={src}
                        alt={alt || fallback}
                        className="h-full w-full object-cover"
                        onError={() => setImgError(true)}
                    />
                ) : (
                    <span aria-label={alt || fallback}>{fallback}</span>
                )}
            </div>
            {status && (
                <span
                    className={cn(
                        "absolute bottom-0 right-0 block rounded-full ring-2 ring-background",
                        size === "sm" && "h-2 w-2",
                        size === "md" && "h-2.5 w-2.5",
                        size === "lg" && "h-3 w-3",
                        status === "online" && "bg-emerald-400",
                        status === "offline" && "bg-zinc-500",
                        status === "busy" && "bg-amber-400",
                    )}
                    aria-label={`Status: ${status}`}
                />
            )}
        </div>
    );
}

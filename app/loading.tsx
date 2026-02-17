import { Skeleton } from "@/components/ui/Skeleton";

export default function Loading() {
    return (
        <div className="min-h-screen flex items-center justify-center bg-background">
            <div className="flex flex-col items-center gap-4">
                <div className="h-8 w-8 rounded-lg bg-foreground/10 flex items-center justify-center animate-pulse">
                    <span className="text-foreground font-bold text-xs">E</span>
                </div>
                <div className="space-y-3 w-48">
                    <Skeleton className="h-2 w-full" />
                    <Skeleton className="h-2 w-3/4 mx-auto" />
                </div>
            </div>
        </div>
    );
}

import { Skeleton } from "@/components/ui/Skeleton";

export default function DashboardLoading() {
    return (
        <div className="max-w-5xl mx-auto space-y-8 animate-fade-in">
            {/* Header skeleton */}
            <div className="space-y-3">
                <Skeleton className="h-8 w-48" />
                <Skeleton className="h-4 w-80" />
            </div>

            {/* Stats skeleton */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {[1, 2, 3].map((i) => (
                    <div key={i} className="rounded-xl border border-border p-6 space-y-3">
                        <Skeleton className="h-4 w-24" />
                        <Skeleton className="h-8 w-16" />
                        <Skeleton className="h-2 w-full" />
                    </div>
                ))}
            </div>

            {/* Cards skeleton */}
            <div className="grid gap-4 md:grid-cols-2">
                {[1, 2, 3, 4].map((i) => (
                    <div key={i} className="rounded-xl border border-border p-6 space-y-4">
                        <div className="flex items-center gap-3">
                            <Skeleton variant="circular" className="h-10 w-10" />
                            <div className="space-y-2 flex-1">
                                <Skeleton className="h-4 w-32" />
                                <Skeleton className="h-3 w-48" />
                            </div>
                        </div>
                        <Skeleton className="h-2 w-full" />
                    </div>
                ))}
            </div>
        </div>
    );
}

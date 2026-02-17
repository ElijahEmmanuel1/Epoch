import Link from "next/link";

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-sans">
      <main className="flex flex-col gap-8 row-start-2 items-center text-center max-w-2xl">
        <h1 className="text-4xl sm:text-6xl font-bold tracking-tight">
          Epoch
        </h1>
        <p className="text-lg sm:text-xl text-muted-foreground font-serif">
          Interactive deep learning platform based on &quot;Understanding Deep Learning&quot; by Simon J.D. Prince.
        </p>

        <div className="flex gap-4 items-center flex-col sm:flex-row">
          <Link href="/dashboard" className="rounded-full border border-solid border-transparent transition-colors flex items-center justify-center bg-foreground text-background gap-2 hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5 cursor-pointer">
            Start Learning
          </Link>
          <Link href="/dashboard" className="rounded-full border border-solid border-black/[.08] dark:border-white/[.145] transition-colors flex items-center justify-center hover:bg-[#f2f2f2] dark:hover:bg-[#1a1a1a] hover:border-transparent text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5 sm:min-w-44 cursor-pointer">
            Read Syllabus
          </Link>
        </div>
      </main>
      <footer className="row-start-3 flex gap-6 flex-wrap items-center justify-center text-muted-foreground text-sm">
        <p>&copy; {new Date().getFullYear()} Epoch Platform</p>
      </footer>
    </div>
  );
}

import type { Metadata } from "next";
import { Noto_Sans, Noto_Serif } from "next/font/google";
import { ThemeProvider } from "@/components/providers/ThemeProvider";
import { ToastProvider } from "@/components/ui/Toast";
import "./globals.css";

const notoSans = Noto_Sans({
  variable: "--font-noto-sans",
  subsets: ["latin"],
  display: "swap",
});

const notoSerif = Noto_Serif({
  variable: "--font-noto-serif",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "Epoch — Interactive Deep Learning",
    template: "%s | Epoch",
  },
  description: "An interactive, code-first platform for mastering deep learning from first principles. Based on Understanding Deep Learning by Simon J.D. Prince.",
  keywords: ["deep learning", "machine learning", "neural networks", "AI", "education", "interactive"],
  authors: [{ name: "Epoch" }],
  openGraph: {
    title: "Epoch — Interactive Deep Learning",
    description: "Master deep learning from first principles with interactive lessons and code.",
    type: "website",
    locale: "en_US",
    siteName: "Epoch",
  },
  twitter: {
    card: "summary_large_image",
    title: "Epoch — Interactive Deep Learning",
    description: "Master deep learning from first principles.",
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" data-theme="dark" suppressHydrationWarning>
      <body
        className={`${notoSans.variable} ${notoSerif.variable} antialiased bg-background text-foreground`}
      >
        <ThemeProvider>
          <ToastProvider>
            {children}
          </ToastProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}

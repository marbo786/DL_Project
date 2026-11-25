import type { Metadata } from 'next';
import localFont from 'next/font/local';
import type { ReactNode } from 'react';
import { ScrollProvider } from '@/components/scroll-provider';
import './globals.css';

const inter = localFont({
  src: './fonts/inter-latin-variable.woff2',
  variable: '--font-inter',
  weight: '100 900',
  display: 'swap',
  preload: false,
});

export const metadata: Metadata = {
  title: 'ForgeryLens | Receipt review',
  description: 'An AI341 research prototype for inspecting receipt images.',
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={inter.variable} suppressHydrationWarning>
      <head><script dangerouslySetInnerHTML={{ __html: "document.documentElement.dataset.enhanced='';try{if(localStorage.getItem('forgerylens-theme')==='light')document.documentElement.dataset.theme='light'}catch{}" }} /></head>
      <body><ScrollProvider>{children}</ScrollProvider></body>
    </html>
  );
}

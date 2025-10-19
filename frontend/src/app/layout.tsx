import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { ThemeProvider } from '@/design-system/utils/theme-provider';
import { QueryProvider } from '@/lib/providers/query-provider';
import { Toaster } from 'react-hot-toast';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'PathwayLens 2.0',
  description: 'Next-generation computational biology platform for multi-omics pathway analysis',
  keywords: [
    'bioinformatics',
    'pathway analysis',
    'multi-omics',
    'gene expression',
    'computational biology',
    'KEGG',
    'Reactome',
    'Gene Ontology',
  ],
  authors: [{ name: 'PathwayLens Team' }],
  creator: 'PathwayLens Team',
  publisher: 'PathwayLens',
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  metadataBase: new URL(process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'),
  openGraph: {
    title: 'PathwayLens 2.0',
    description: 'Next-generation computational biology platform for multi-omics pathway analysis',
    url: '/',
    siteName: 'PathwayLens',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'PathwayLens 2.0',
      },
    ],
    locale: 'en_US',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'PathwayLens 2.0',
    description: 'Next-generation computational biology platform for multi-omics pathway analysis',
    images: ['/og-image.png'],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  verification: {
    google: process.env.GOOGLE_SITE_VERIFICATION,
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <QueryProvider>
            {children}
            <Toaster
              position="top-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: 'hsl(var(--card))',
                  color: 'hsl(var(--card-foreground))',
                  border: '1px solid hsl(var(--border))',
                },
                success: {
                  iconTheme: {
                    primary: 'hsl(var(--success))',
                    secondary: 'hsl(var(--success-foreground))',
                  },
                },
                error: {
                  iconTheme: {
                    primary: 'hsl(var(--destructive))',
                    secondary: 'hsl(var(--destructive-foreground))',
                  },
                },
              }}
            />
          </QueryProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}

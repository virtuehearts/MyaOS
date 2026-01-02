import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'MyaOS',
  description: 'OS-like interface for the Mya companion'
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

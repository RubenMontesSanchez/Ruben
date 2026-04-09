import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "3D Generator",
  description: "Generate 3D models from text or images — print locally",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  );
}

import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BRD4 QSAR Predictor",
  description: "Ligand-based BRD4 pIC50 predictions with applicability-domain checks.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}

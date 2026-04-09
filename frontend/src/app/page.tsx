"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import GeneratorForm from "@/components/GeneratorForm";
import DownloadPanel from "@/components/DownloadPanel";

// Load Three.js only on client (no SSR)
const ModelViewer = dynamic(() => import("@/components/ModelViewer"), { ssr: false });

export default function Home() {
  const [modelUrl, setModelUrl] = useState<string | null>(null);
  const [completedJobId, setCompletedJobId] = useState<string | null>(null);

  const handleComplete = (jobId: string, outputUrl: string) => {
    setCompletedJobId(jobId);
    setModelUrl(outputUrl);
  };

  return (
    <main className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-slate-800 px-6 py-4 flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-brand-500 flex items-center justify-center">
          <svg viewBox="0 0 24 24" fill="none" className="w-5 h-5 text-white" stroke="currentColor" strokeWidth={2}>
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
          </svg>
        </div>
        <h1 className="text-lg font-bold text-white">3D Generator</h1>
        <span className="text-xs text-slate-500 ml-1">Local &amp; Free</span>
      </header>

      {/* Main layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left panel — form */}
        <aside className="w-96 flex-shrink-0 border-r border-slate-800 p-6 overflow-y-auto">
          <GeneratorForm onComplete={handleComplete} />

          {completedJobId && (
            <div className="mt-6">
              <DownloadPanel jobId={completedJobId} />
            </div>
          )}
        </aside>

        {/* Right panel — 3D viewer */}
        <section className="flex-1 relative p-6">
          <div className="w-full h-full relative">
            <ModelViewer url={modelUrl} />
          </div>

          {modelUrl && (
            <p className="absolute bottom-8 left-1/2 -translate-x-1/2 text-xs text-slate-500">
              Arrastra para rotar · Scroll para zoom
            </p>
          )}
        </section>
      </div>
    </main>
  );
}

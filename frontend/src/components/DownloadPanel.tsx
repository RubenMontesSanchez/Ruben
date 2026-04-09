"use client";

import { downloadUrl } from "@/lib/api";

interface Props {
  jobId: string;
}

const FORMATS = [
  { fmt: "stl" as const, label: "STL", desc: "Para impresora 3D" },
  { fmt: "obj" as const, label: "OBJ", desc: "Para edición 3D" },
  { fmt: "glb" as const, label: "GLB", desc: "Para visualización" },
];

export default function DownloadPanel({ jobId }: Props) {
  return (
    <div className="flex flex-col gap-2">
      <p className="text-sm text-slate-400 mb-1">Descargar modelo</p>
      <div className="flex gap-3">
        {FORMATS.map(({ fmt, label, desc }) => (
          <a
            key={fmt}
            href={downloadUrl(jobId, fmt)}
            download
            className="flex flex-col items-center gap-1 px-4 py-3 glass rounded-xl hover:bg-white/10 transition-colors group"
          >
            <span className="text-brand-500 font-bold text-sm group-hover:text-brand-400">
              .{label}
            </span>
            <span className="text-xs text-slate-500">{desc}</span>
          </a>
        ))}
      </div>
    </div>
  );
}

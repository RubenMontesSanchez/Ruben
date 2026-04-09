"use client";

interface Props {
  progress: number;
  status: string;
}

const STATUS_LABELS: Record<string, string> = {
  pending: "En cola...",
  processing: "Generando modelo 3D...",
  completed: "Completado",
  error: "Error",
};

export default function ProgressBar({ progress, status }: Props) {
  const isError = status === "error";

  return (
    <div className="w-full space-y-2">
      <div className="flex justify-between text-sm text-slate-400">
        <span>{STATUS_LABELS[status] ?? status}</span>
        <span>{progress}%</span>
      </div>
      <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
        <div
          className={`h-2.5 rounded-full transition-all duration-500 ${
            isError ? "bg-red-500" : "bg-brand-500"
          }`}
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
}

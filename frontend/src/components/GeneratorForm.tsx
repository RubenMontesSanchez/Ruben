"use client";

import { useState, useRef } from "react";
import { generateFromText, generateFromImage, pollStatus, JobStatus, TextOptions } from "@/lib/api";
import ProgressBar from "./ProgressBar";

interface Props {
  onComplete: (jobId: string, outputUrl: string) => void;
}

type Tab = "text" | "image";
type Quality = "fast" | "normal" | "high";

const RESOLUTIONS = [
  { value: 128, label: "Rápida", time: "~10s" },
  { value: 256, label: "Normal", time: "~30s" },
  { value: 384, label: "Alta",   time: "~60s" },
];

const QUALITIES: { value: Quality; label: string; time: string }[] = [
  { value: "fast",   label: "Rápida", time: "~20s" },
  { value: "normal", label: "Normal", time: "~45s" },
  { value: "high",   label: "Alta",   time: "~90s" },
];

function Toggle({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!value)}
      className={`relative w-11 h-6 rounded-full transition-colors ${value ? "bg-brand-500" : "bg-slate-700"}`}
    >
      <span
        className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${
          value ? "translate-x-5" : "translate-x-0"
        }`}
      />
    </button>
  );
}

function SegmentedControl<T extends string | number>({
  options,
  value,
  onChange,
}: {
  options: { value: T; label: string; time: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div className="flex gap-1 p-1 bg-slate-800 rounded-lg">
      {options.map((o) => (
        <button
          key={String(o.value)}
          onClick={() => onChange(o.value)}
          className={`flex-1 py-1.5 rounded-md text-xs font-medium transition-all ${
            value === o.value ? "bg-brand-500 text-white" : "text-slate-400 hover:text-white"
          }`}
        >
          {o.label}
          <span className="block text-[10px] opacity-70">{o.time}</span>
        </button>
      ))}
    </div>
  );
}

export default function GeneratorForm({ onComplete }: Props) {
  const [tab, setTab]               = useState<Tab>("text");
  const [prompt, setPrompt]         = useState("");
  const [imageFile, setImageFile]   = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [loading, setLoading]       = useState(false);
  const [job, setJob]               = useState<JobStatus | null>(null);
  const [error, setError]           = useState<string | null>(null);

  // Image options
  const [resolution, setResolution] = useState(256);
  const [removeBg, setRemoveBg]     = useState(true);
  const [enhance, setEnhance]       = useState(false);

  // Text options
  const [quality, setQuality]       = useState<Quality>("normal");

  const fileRef = useRef<HTMLInputElement>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
  };

  const handleSubmit = async () => {
    setError(null);
    setJob(null);
    setLoading(true);
    try {
      let jobId: string;
      if (tab === "text") {
        if (!prompt.trim()) throw new Error("Escribe una descripción");
        jobId = await generateFromText(prompt.trim(), { quality });
      } else {
        if (!imageFile) throw new Error("Selecciona una imagen");
        jobId = await generateFromImage(imageFile, { resolution, removeBg, enhance });
      }

      await new Promise<void>((resolve, reject) => {
        const interval = setInterval(async () => {
          try {
            const status = await pollStatus(jobId);
            setJob(status);
            if (status.status === "completed" && status.output) {
              clearInterval(interval);
              onComplete(jobId, status.output);
              resolve();
            } else if (status.status === "error") {
              clearInterval(interval);
              reject(new Error(status.error ?? "Error desconocido"));
            }
          } catch (err) {
            clearInterval(interval);
            reject(err);
          }
        }, 1000);
      });
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Error desconocido");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-5">
      {/* Tabs */}
      <div className="flex gap-1 p-1 glass rounded-xl w-fit">
        {(["text", "image"] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-5 py-2 rounded-lg text-sm font-medium transition-all ${
              tab === t ? "bg-brand-500 text-white shadow" : "text-slate-400 hover:text-white"
            }`}
          >
            {t === "text" ? "Texto" : "Imagen"}
          </button>
        ))}
      </div>

      {/* Input area */}
      {tab === "text" ? (
        <div className="flex flex-col gap-2">
          <label className="text-sm text-slate-400">Describe el objeto</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Ej: una silla moderna de diseño minimalista"
            rows={4}
            className="w-full bg-slate-800/60 border border-slate-700 rounded-xl px-4 py-3 text-slate-200 placeholder-slate-500 resize-none focus:outline-none focus:border-brand-500 transition-colors"
          />
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          <label className="text-sm text-slate-400">Sube una imagen</label>
          <div
            onClick={() => fileRef.current?.click()}
            className="w-full h-40 border-2 border-dashed border-slate-700 rounded-xl flex items-center justify-center cursor-pointer hover:border-brand-500 transition-colors overflow-hidden"
          >
            {imagePreview ? (
              <img src={imagePreview} alt="preview" className="object-contain h-full" />
            ) : (
              <p className="text-slate-500 text-sm">Haz clic para seleccionar imagen</p>
            )}
          </div>
          <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={handleImageChange} />
        </div>
      )}

      {/* Options panel */}
      <div className="flex flex-col gap-4 p-4 glass rounded-xl">
        <p className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Opciones</p>

        {tab === "image" ? (
          <>
            <div className="flex flex-col gap-1.5">
              <label className="text-sm text-slate-300">Resolución del modelo</label>
              <SegmentedControl options={RESOLUTIONS} value={resolution} onChange={setResolution} />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-300">Eliminar fondo</p>
                <p className="text-xs text-slate-500">Aísla el objeto automáticamente</p>
              </div>
              <Toggle value={removeBg} onChange={setRemoveBg} />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-300">Mejorar imagen</p>
                <p className="text-xs text-slate-500">Aumenta contraste y nitidez</p>
              </div>
              <Toggle value={enhance} onChange={setEnhance} />
            </div>
          </>
        ) : (
          <div className="flex flex-col gap-1.5">
            <label className="text-sm text-slate-300">Calidad de generación</label>
            <SegmentedControl options={QUALITIES} value={quality} onChange={setQuality} />
          </div>
        )}
      </div>

      {/* Progress */}
      {job && (job.status === "pending" || job.status === "processing") && (
        <ProgressBar progress={job.progress} status={job.status} />
      )}

      {/* Error */}
      {error && (
        <p className="text-red-400 text-sm bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-2">
          {error}
        </p>
      )}

      {/* Submit */}
      <button
        onClick={handleSubmit}
        disabled={loading}
        className="w-full py-3 rounded-xl font-semibold text-white bg-brand-500 hover:bg-brand-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all active:scale-95"
      >
        {loading ? "Generando..." : "Generar modelo 3D"}
      </button>
    </div>
  );
}

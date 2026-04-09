"use client";

import { useState, useRef } from "react";
import { generateFromText, generateFromImage, pollStatus, JobStatus } from "@/lib/api";
import ProgressBar from "./ProgressBar";

interface Props {
  onComplete: (jobId: string, outputUrl: string) => void;
}

type Tab = "text" | "image";

export default function GeneratorForm({ onComplete }: Props) {
  const [tab, setTab] = useState<Tab>("text");
  const [prompt, setPrompt] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
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
        jobId = await generateFromText(prompt.trim());
      } else {
        if (!imageFile) throw new Error("Selecciona una imagen");
        jobId = await generateFromImage(imageFile);
      }

      // Poll for completion
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
              tab === t
                ? "bg-brand-500 text-white shadow"
                : "text-slate-400 hover:text-white"
            }`}
          >
            {t === "text" ? "Texto" : "Imagen"}
          </button>
        ))}
      </div>

      {/* Input */}
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
            className="w-full h-40 border-2 border-dashed border-slate-700 rounded-xl flex items-center justify-center cursor-pointer hover:border-brand-500 transition-colors overflow-hidden relative"
          >
            {imagePreview ? (
              <img src={imagePreview} alt="preview" className="object-contain h-full" />
            ) : (
              <p className="text-slate-500 text-sm">Haz clic para seleccionar imagen</p>
            )}
          </div>
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleImageChange}
          />
        </div>
      )}

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

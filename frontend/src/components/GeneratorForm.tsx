"use client";

import { useState, useRef } from "react";
import {
  generateFromText, generateFromImage, generateFace, generateBody,
  analyzeImage, pollStatus,
  JobStatus, TextOptions, FaceOptions, BodyOptions,
} from "@/lib/api";
import ProgressBar from "./ProgressBar";

interface Props {
  onComplete: (jobId: string, outputUrl: string) => void;
}

type Tab = "text" | "image" | "human";
type Quality = "fast" | "normal" | "high";
type HumanMode = "face" | "body";
type BodyQuality = "fast" | "quality";

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

/* ── Image uploader shared between tabs ──────────────────────────────────── */
function ImageUploader({
  preview,
  onFile,
  analyzing,
  detection,
}: {
  preview: string | null;
  onFile: (e: React.ChangeEvent<HTMLInputElement>) => void;
  analyzing: boolean;
  detection: { label: string; label_es: string; confidence: number; error?: string } | null;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  return (
    <div className="flex flex-col gap-2">
      <label className="text-sm text-slate-400">Sube una imagen</label>
      <div
        onClick={() => fileRef.current?.click()}
        className="w-full h-40 border-2 border-dashed border-slate-700 rounded-xl flex items-center justify-center cursor-pointer hover:border-brand-500 transition-colors overflow-hidden"
      >
        {preview ? (
          <img src={preview} alt="preview" className="object-contain h-full" />
        ) : (
          <p className="text-slate-500 text-sm">Haz clic para seleccionar imagen</p>
        )}
      </div>
      <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={onFile} />
      {analyzing && (
        <p className="text-xs text-slate-400 animate-pulse">Analizando imagen...</p>
      )}
      {detection && !analyzing && (
        <div className="flex items-center gap-2 px-3 py-1.5 glass rounded-lg w-fit">
          <span className="text-xs text-slate-400">Detectado:</span>
          {detection.error ? (
            <span className="text-xs text-amber-400">{detection.error}</span>
          ) : (
            <>
              <span className="text-xs font-semibold text-brand-400 capitalize">{detection.label_es}</span>
              <span className="text-xs text-slate-500">{Math.round(detection.confidence * 100)}%</span>
            </>
          )}
        </div>
      )}
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
  const [detection, setDetection]   = useState<{ label: string; label_es: string; confidence: number; error?: string } | null>(null);
  const [analyzing, setAnalyzing]   = useState(false);

  // Image options
  const [resolution, setResolution] = useState(256);
  const [removeBg, setRemoveBg]     = useState(true);
  const [enhance, setEnhance]       = useState(false);
  const [pipeline, setPipeline]     = useState<"standard" | "advanced">("standard");

  // Text options
  const [quality, setQuality]       = useState<Quality>("normal");

  // Human reconstruction options
  const [humanMode, setHumanMode]   = useState<HumanMode>("face");
  const [addBustBase, setAddBustBase] = useState(true);
  const [bodyQuality, setBodyQuality] = useState<BodyQuality>("fast");
  const [addBase, setAddBase]       = useState(true);

  const handleImageChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    setDetection(null);
    setAnalyzing(true);
    try {
      const result = await analyzeImage(file);
      setDetection(result);
    } catch {
      // silent — detection is optional
    } finally {
      setAnalyzing(false);
    }
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

      } else if (tab === "image") {
        if (!imageFile) throw new Error("Selecciona una imagen");
        jobId = await generateFromImage(imageFile, { resolution, removeBg, enhance, pipeline });

      } else {
        // human tab
        if (!imageFile) throw new Error("Selecciona una imagen");
        if (humanMode === "face") {
          jobId = await generateFace(imageFile, { addBustBase });
        } else {
          jobId = await generateBody(imageFile, { mode: bodyQuality, addBase });
        }
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
        {(["text", "image", "human"] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-5 py-2 rounded-lg text-sm font-medium transition-all ${
              tab === t ? "bg-brand-500 text-white shadow" : "text-slate-400 hover:text-white"
            }`}
          >
            {t === "text" ? "Texto" : t === "image" ? "Imagen" : "Humano"}
          </button>
        ))}
      </div>

      {/* ── Text tab ───────────────────────────────────────────────────────── */}
      {tab === "text" && (
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
      )}

      {/* ── Image tab ──────────────────────────────────────────────────────── */}
      {tab === "image" && (
        <ImageUploader
          preview={imagePreview}
          onFile={handleImageChange}
          analyzing={analyzing}
          detection={detection}
        />
      )}

      {/* ── Human tab ──────────────────────────────────────────────────────── */}
      {tab === "human" && (
        <div className="flex flex-col gap-3">
          {/* Mode selector */}
          <div className="flex gap-1 p-1 bg-slate-800 rounded-lg">
            <button
              onClick={() => setHumanMode("face")}
              className={`flex-1 py-2 rounded-md text-xs font-medium transition-all ${
                humanMode === "face" ? "bg-brand-500 text-white" : "text-slate-400 hover:text-white"
              }`}
            >
              Cara / Busto
              <span className="block text-[10px] opacity-70">MediaPipe · ~15s</span>
            </button>
            <button
              onClick={() => setHumanMode("body")}
              className={`flex-1 py-2 rounded-md text-xs font-medium transition-all ${
                humanMode === "body" ? "bg-brand-500 text-white" : "text-slate-400 hover:text-white"
              }`}
            >
              Cuerpo completo
              <span className="block text-[10px] opacity-70">MediaPipe Pose · ~25s</span>
            </button>
          </div>

          {/* Tips */}
          <div className="px-3 py-2 rounded-lg bg-slate-800/60 border border-slate-700/50">
            {humanMode === "face" ? (
              <p className="text-[11px] text-slate-400 leading-snug">
                Usa una foto <strong className="text-slate-300">frontal, bien iluminada</strong> con la cara centrada.
                Sin gafas de sol. Cuanto más cerca, más detalle.
              </p>
            ) : (
              <p className="text-[11px] text-slate-400 leading-snug">
                Foto de <strong className="text-slate-300">cuerpo entero</strong>, de frente,
                con los brazos ligeramente separados del cuerpo. Fondo liso recomendado.
              </p>
            )}
          </div>

          <ImageUploader
            preview={imagePreview}
            onFile={handleImageChange}
            analyzing={analyzing}
            detection={detection}
          />
        </div>
      )}

      {/* ── Options panel ──────────────────────────────────────────────────── */}
      <div className="flex flex-col gap-4 p-4 glass rounded-xl">
        <p className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Opciones</p>

        {tab === "image" && (
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

            <div className="flex flex-col gap-1.5 pt-1 border-t border-slate-700/50">
              <label className="text-sm text-slate-300">Pipeline de reconstrucción</label>
              <div className="flex gap-1 p-1 bg-slate-800 rounded-lg">
                <button
                  onClick={() => setPipeline("standard")}
                  className={`flex-1 py-1.5 rounded-md text-xs font-medium transition-all ${
                    pipeline === "standard" ? "bg-brand-500 text-white" : "text-slate-400 hover:text-white"
                  }`}
                >
                  Estándar
                  <span className="block text-[10px] opacity-70">TripoSR · ~30s</span>
                </button>
                <button
                  onClick={() => setPipeline("advanced")}
                  className={`flex-1 py-1.5 rounded-md text-xs font-medium transition-all ${
                    pipeline === "advanced" ? "bg-brand-500 text-white" : "text-slate-400 hover:text-white"
                  }`}
                >
                  Avanzada
                  <span className="block text-[10px] opacity-70">Zero123++ + TripoSR · ~3min</span>
                </button>
              </div>
              {pipeline === "advanced" && (
                <p className="text-[11px] text-amber-400/80 leading-snug">
                  Primera vez descarga ~5.5 GB. Gestión VRAM secuencial para no saturar la GPU.
                </p>
              )}
            </div>
          </>
        )}

        {tab === "text" && (
          <div className="flex flex-col gap-1.5">
            <label className="text-sm text-slate-300">Calidad de generación</label>
            <SegmentedControl options={QUALITIES} value={quality} onChange={setQuality} />
          </div>
        )}

        {tab === "human" && humanMode === "face" && (
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-300">Añadir base de busto</p>
              <p className="text-xs text-slate-500">Cuello y base cilíndrica inferior</p>
            </div>
            <Toggle value={addBustBase} onChange={setAddBustBase} />
          </div>
        )}

        {tab === "human" && humanMode === "body" && (
          <>
            <div className="flex flex-col gap-1.5">
              <label className="text-sm text-slate-300">Modo de reconstrucción</label>
              <div className="flex gap-1 p-1 bg-slate-800 rounded-lg">
                <button
                  onClick={() => setBodyQuality("fast")}
                  className={`flex-1 py-1.5 rounded-md text-xs font-medium transition-all ${
                    bodyQuality === "fast" ? "bg-brand-500 text-white" : "text-slate-400 hover:text-white"
                  }`}
                >
                  Rápido
                  <span className="block text-[10px] opacity-70">Landmarks · ~25s</span>
                </button>
                <button
                  onClick={() => setBodyQuality("quality")}
                  className={`flex-1 py-1.5 rounded-md text-xs font-medium transition-all ${
                    bodyQuality === "quality" ? "bg-brand-500 text-white" : "text-slate-400 hover:text-white"
                  }`}
                >
                  Calidad
                  <span className="block text-[10px] opacity-70">PIFuHD · ~2min</span>
                </button>
              </div>
              {bodyQuality === "quality" && (
                <p className="text-[11px] text-amber-400/80 leading-snug">
                  Requiere PIFuHD instalado en backend/PIFuHD/. Si no está disponible, usará el modo rápido.
                </p>
              )}
            </div>

            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-300">Base de peana</p>
                <p className="text-xs text-slate-500">Disco plano para imprimir en 3D</p>
              </div>
              <Toggle value={addBase} onChange={setAddBase} />
            </div>
          </>
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

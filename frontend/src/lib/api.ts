import axios from "axios";

const BASE = "/api";

export interface JobStatus {
  status: "pending" | "processing" | "completed" | "error";
  progress: number;
  output: string | null;
  error: string | null;
}

export interface ImageOptions {
  resolution: number;
  removeBg: boolean;
  enhance: boolean;
  pipeline: "standard" | "advanced";
}

export interface TextOptions {
  quality: "fast" | "normal" | "high";
}

export interface FaceOptions {
  addBustBase: boolean;
}

export interface BodyOptions {
  mode: "fast" | "quality";
  addBase: boolean;
}

export async function generateFromText(prompt: string, options: TextOptions): Promise<string> {
  const form = new FormData();
  form.append("prompt", prompt);
  form.append("quality", options.quality);
  const res = await axios.post(`${BASE}/generate/text`, form);
  return res.data.job_id;
}

export async function generateFromImage(file: File, options: ImageOptions): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  form.append("resolution", options.resolution.toString());
  form.append("remove_bg", options.removeBg.toString());
  form.append("enhance", options.enhance.toString());
  form.append("pipeline", options.pipeline);
  const res = await axios.post(`${BASE}/generate/image`, form);
  return res.data.job_id;
}

export async function generateFace(file: File, options: FaceOptions): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  form.append("add_bust_base", options.addBustBase.toString());
  const res = await axios.post(`${BASE}/generate/face`, form);
  return res.data.job_id;
}

export async function generateBody(file: File, options: BodyOptions): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  form.append("mode", options.mode);
  form.append("add_base", options.addBase.toString());
  const res = await axios.post(`${BASE}/generate/body`, form);
  return res.data.job_id;
}

export async function analyzeImage(file: File): Promise<{ label: string; label_es: string; confidence: number }> {
  const form = new FormData();
  form.append("file", file);
  const res = await axios.post(`${BASE}/analyze`, form);
  return res.data;
}

export async function pollStatus(jobId: string): Promise<JobStatus> {
  const res = await axios.get(`${BASE}/status/${jobId}`);
  return res.data;
}

export function downloadUrl(jobId: string, fmt: "stl" | "obj" | "glb"): string {
  return `${BASE}/download/${jobId}/${fmt}`;
}

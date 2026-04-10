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
}

export interface TextOptions {
  quality: "fast" | "normal" | "high";
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
  const res = await axios.post(`${BASE}/generate/image`, form);
  return res.data.job_id;
}

export async function pollStatus(jobId: string): Promise<JobStatus> {
  const res = await axios.get(`${BASE}/status/${jobId}`);
  return res.data;
}

export function downloadUrl(jobId: string, fmt: "stl" | "obj" | "glb"): string {
  return `${BASE}/download/${jobId}/${fmt}`;
}

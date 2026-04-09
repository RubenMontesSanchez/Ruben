import axios from "axios";

const BASE = "/api";

export interface JobStatus {
  status: "pending" | "processing" | "completed" | "error";
  progress: number;
  output: string | null;
  error: string | null;
}

export async function generateFromText(prompt: string): Promise<string> {
  const form = new FormData();
  form.append("prompt", prompt);
  const res = await axios.post(`${BASE}/generate/text`, form);
  return res.data.job_id;
}

export async function generateFromImage(file: File): Promise<string> {
  const form = new FormData();
  form.append("file", file);
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

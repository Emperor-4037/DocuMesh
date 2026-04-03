import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

// A demo JWT for development — in production the gateway validates real tokens
const DEMO_TOKEN = import.meta.env.VITE_DEMO_TOKEN ?? 'demo-token';

const api = axios.create({
  baseURL: API_BASE,
  headers: { Authorization: `Bearer ${DEMO_TOKEN}` },
  timeout: 60_000,
});

export const paraphrase = (text, tone = 'neutral') =>
  api.post('/api/paraphrase', { text, tone });

export const grammar = (text) =>
  api.post('/api/grammar', { text });

export const simplify = (text, reading_level = 'middle school') =>
  api.post('/api/simplify', { text, reading_level });

export const tone = (text, target_tone = 'professional') =>
  api.post('/api/tone', { text, target_tone });

export const summarize = (text, max_length = 150) =>
  api.post('/api/summarize', { text, max_length });

export const ragQuery = (query, top_k = 5) =>
  api.post('/api/rag/query', { query, top_k });

export const ragIngest = (file, onProgress) => {
  const form = new FormData();
  form.append('file', file);
  return api.post('/api/rag/ingest', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => onProgress && onProgress(Math.round((e.loaded * 100) / e.total)),
  });
};

export const ingestStatus = (taskId) =>
  api.get(`/api/rag/ingest/status/${taskId}`);

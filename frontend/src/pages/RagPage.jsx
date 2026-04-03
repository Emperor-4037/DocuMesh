import { useState, useRef } from 'react';
import { Database, Upload, SendHorizonal, FileText, CheckCircle2, X } from 'lucide-react';
import { ragIngest, ragQuery } from '../api';
import { useToast } from '../context/ToastContext';
import { Spinner, FormGroup } from '../components/ui';

export default function RagPage() {
  // Upload state
  const [file, setFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const fileInputRef = useRef();

  // Chat state
  const [messages, setMessages] = useState([
    { role: 'bot', text: 'Hello! Upload a document and I\'ll answer questions about it using retrieval-augmented generation.' }
  ]);
  const [query, setQuery] = useState('');
  const [querying, setQuerying] = useState(false);
  const [topK, setTopK] = useState(5);
  const messagesEndRef = useRef();

  const toast = useToast();

  // --- Upload ---
  const handleFileChange = (f) => { if (f) setFile(f); };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setUploadProgress(0);
    try {
      const { data } = await ragIngest(file, setUploadProgress);
      setUploadedFiles((prev) => [...prev, { name: file.name, taskId: data.task_id }]);
      setFile(null);
      toast(`"${file.name}" queued for ingestion (task: ${data.task_id?.slice(0, 8)}…)`, 'success');
    } catch (err) {
      toast(err.response?.data?.detail ?? 'Upload failed', 'error');
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  // --- Query ---
  const handleQuery = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    const q = query.trim();
    setQuery('');
    setMessages((prev) => [...prev, { role: 'user', text: q }]);
    setQuerying(true);
    try {
      const { data } = await ragQuery(q, topK);
      setMessages((prev) => [
        ...prev,
        { role: 'bot', text: data.answer, sources: data.sources }
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'bot', text: '⚠️ ' + (err.response?.data?.detail ?? 'Query failed. Make sure documents have been ingested.') }
      ]);
    } finally {
      setQuerying(false);
      setTimeout(() => messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 100);
    }
  };

  return (
    <PageWrapper>
      <div className="page-header">
        <h1>Document Assistant</h1>
        <p>Upload your documents and chat with them using RAG — Retrieval-Augmented Generation.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.5fr', gap: 20, alignItems: 'start' }}>
        {/* Upload Panel */}
        <div className="card">
          <div className="card-header">
            <div className="card-icon" style={{ background: 'rgba(124,58,237,0.15)', color: 'var(--clr-accent)' }}>
              <Upload size={18} />
            </div>
            <div>
              <div className="card-title">Ingest Documents</div>
              <div className="card-desc">PDF, TXT, Markdown supported</div>
            </div>
          </div>

          <div
            id="rag-drop-zone"
            className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="upload-zone-icon" style={{ display: 'flex', justifyContent: 'center' }}>
              <FileText size={36} />
            </div>
            <p><strong>Click to browse</strong> or drag & drop</p>
            <p className="text-xs mt-2" style={{ color: 'var(--clr-muted)' }}>PDF, TXT, MD</p>
            <input
              ref={fileInputRef}
              type="file"
              id="rag-file-input"
              accept=".pdf,.txt,.md"
              style={{ display: 'none' }}
              onChange={(e) => handleFileChange(e.target.files[0])}
            />
          </div>

          {file && (
            <div className="card-glass mt-3" style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px' }}>
              <FileText size={16} style={{ color: 'var(--clr-accent)', flexShrink: 0 }} />
              <span className="truncate text-sm" style={{ flex: 1 }}>{file.name}</span>
              <span className="text-xs text-muted">({(file.size / 1024).toFixed(1)} KB)</span>
              <button className="btn btn-ghost btn-icon btn-sm" onClick={() => setFile(null)}><X size={14} /></button>
            </div>
          )}

          {uploading && (
            <div className="mt-3">
              <div style={{ height: 4, background: 'rgba(255,255,255,0.07)', borderRadius: 4, overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  background: 'linear-gradient(90deg, var(--clr-primary), var(--clr-accent))',
                  width: `${uploadProgress}%`,
                  transition: 'width 0.2s'
                }} />
              </div>
              <div className="text-xs text-muted mt-2">{uploadProgress}% uploaded</div>
            </div>
          )}

          <button
            id="rag-upload-btn"
            className="btn btn-primary w-full mt-3"
            onClick={handleUpload}
            disabled={!file || uploading}
          >
            {uploading ? <Spinner /> : <Upload size={15} />}
            {uploading ? 'Uploading…' : 'Ingest Document'}
          </button>

          {uploadedFiles.length > 0 && (
            <div className="mt-3">
              <div className="form-label mb-2">Ingested Files</div>
              {uploadedFiles.map((f, i) => (
                <div key={i} className="flex items-center gap-2 mt-2">
                  <CheckCircle2 size={14} style={{ color: 'var(--clr-success)', flexShrink: 0 }} />
                  <span className="text-sm truncate">{f.name}</span>
                  <span className="badge badge-green" style={{ marginLeft: 'auto' }}>queued</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Chat Panel */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="card-header">
            <div className="card-icon" style={{ background: 'rgba(124,58,237,0.15)', color: 'var(--clr-accent)' }}>
              <Database size={18} />
            </div>
            <div style={{ flex: 1 }}>
              <div className="card-title">Ask Your Documents</div>
              <div className="card-desc">Hybrid search + cross-encoder reranking + LLM answer</div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted">Top-K</span>
              <select
                id="rag-topk-select"
                className="input"
                style={{ width: 64, padding: '4px 8px', minHeight: 'unset', fontSize: '0.8rem' }}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
              >
                {[3, 5, 8, 10].map(k => <option key={k} value={k}>{k}</option>)}
              </select>
            </div>
          </div>

          <div className="chat-messages">
            {messages.map((m, i) => (
              <div key={i} className={`chat-msg ${m.role}`}>
                <div className={`chat-avatar ${m.role === 'bot' ? 'bot' : 'user-av'}`}>
                  {m.role === 'bot' ? 'AI' : 'U'}
                </div>
                <div>
                  <div className={`chat-bubble ${m.role === 'bot' ? 'bot' : 'user'}`}>{m.text}</div>
                  {m.sources?.length > 0 && (
                    <div className="chat-source-tag">
                      Sources:
                      {m.sources.map((s, j) => (
                        <span key={j} className="badge badge-purple">{s.source}{s.page != null ? ` p.${s.page}` : ''}</span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {querying && (
              <div className="chat-msg">
                <div className="chat-avatar bot">AI</div>
                <div className="chat-bubble bot flex items-center gap-2"><Spinner /> Searching…</div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="divider" />

          <form className="chat-input-row" onSubmit={handleQuery}>
            <input
              id="rag-chat-input"
              className="input"
              type="text"
              placeholder="Ask a question about your documents…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              style={{ minHeight: 'unset', padding: '10px 14px' }}
            />
            <button id="rag-send-btn" className="btn btn-primary" type="submit" disabled={querying || !query.trim()}>
              <SendHorizonal size={16} />
            </button>
          </form>
        </div>
      </div>
    </PageWrapper>
  );
}


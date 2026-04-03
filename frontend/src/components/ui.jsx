import { useState, useCallback } from 'react';
import clsx from 'clsx';
import { Copy, Check } from 'lucide-react';

export function ResultBox({ text }) {
  const [copied, setCopied] = useState(false);

  const copy = useCallback(() => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    });
  }, [text]);

  if (!text) return null;
  return (
    <div style={{ position: 'relative' }}>
      <div className="result-box">{text}</div>
      <button
        className="btn btn-ghost btn-icon btn-sm"
        onClick={copy}
        title="Copy to clipboard"
        style={{ position: 'absolute', top: 8, right: 8 }}
      >
        {copied ? <Check size={14} style={{ color: 'var(--clr-success)' }} /> : <Copy size={14} />}
      </button>
    </div>
  );
}

export function Spinner() {
  return <span className="spinner" />;
}

export function FormGroup({ label, children }) {
  return (
    <div className="form-group">
      {label && <label className="form-label">{label}</label>}
      {children}
    </div>
  );
}

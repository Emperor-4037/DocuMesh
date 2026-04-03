import { useState } from 'react';
import PageWrapper from '../components/PageWrapper';
import { AlignLeft, ArrowRight } from 'lucide-react';
import { summarize } from '../api';
import { useToast } from '../context/ToastContext';
import { ResultBox, Spinner, FormGroup } from '../components/ui';

export default function SummarizePage() {
  const [text, setText] = useState('');
  const [maxLength, setMaxLength] = useState(150);
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);
  const toast = useToast();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim()) return;
    setLoading(true);
    setResult('');
    try {
      const { data } = await summarize(text, maxLength);
      setResult(data.summary);
      toast('Summary generated!', 'success');
    } catch (err) {
      toast(err.response?.data?.detail ?? 'Summarization failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageWrapper>
      <div className="page-header">
        <h1>Summarize</h1>
        <p>Condense long documents into crisp, accurate summaries using BART.</p>
      </div>

      <div className="card" style={{ maxWidth: 760 }}>
        <div className="card-header">
          <div className="card-icon" style={{ background: 'rgba(239,68,68,0.1)', color: 'var(--clr-error)' }}>
            <AlignLeft size={20} />
          </div>
          <div>
            <div className="card-title">Text Summarizer</div>
            <div className="card-desc">facebook/bart-large-cnn — abstractive summarization</div>
          </div>
        </div>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <FormGroup label="Input Text (paste a long document, article, or notes)">
            <textarea
              id="summarize-input"
              className="input"
              placeholder="Paste your long text here…"
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={7}
            />
          </FormGroup>

          <FormGroup label={`Max Summary Length: ${maxLength} tokens`}>
            <input
              id="summarize-length-slider"
              type="range"
              min={50}
              max={400}
              step={10}
              value={maxLength}
              onChange={(e) => setMaxLength(Number(e.target.value))}
              style={{ accentColor: 'var(--clr-primary)', width: '100%', cursor: 'pointer' }}
            />
          </FormGroup>

          <div style={{ display: 'flex', gap: 10 }}>
            <button id="summarize-submit-btn" className="btn btn-primary" type="submit" disabled={loading || !text.trim()}>
              {loading ? <Spinner /> : <ArrowRight size={16} />}
              {loading ? 'Summarizing…' : 'Summarize'}
            </button>
            <button className="btn btn-secondary" type="button" onClick={() => { setText(''); setResult(''); }}>
              Clear
            </button>
          </div>
        </form>

        {result && (
          <div className="mt-4">
            <div className="form-label mb-2">Summary</div>
            <ResultBox text={result} />
            <div className="text-xs text-muted mt-2">
              {result.split(' ').length} words · {result.length} characters
            </div>
          </div>
        )}
      </div>
    </PageWrapper>
  );
}


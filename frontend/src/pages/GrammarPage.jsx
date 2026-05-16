import { useState } from 'react';
import PageWrapper from '../components/PageWrapper';
import { ShieldCheck, ArrowRight } from 'lucide-react';
import { grammar } from '../api';
import { useToast } from '../context/ToastContext';
import { ResultBox, Spinner, FormGroup } from '../components/ui';

export default function GrammarPage() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const toast = useToast();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim()) return;
    setLoading(true);
    setResult(null);
    try {
      const { data } = await grammar(text);
      setResult(data);
      toast('Grammar check complete!', 'success');
    } catch (err) {
      toast(err.response?.data?.detail ?? 'Grammar check failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageWrapper>
      <div className="page-header">
        <h1>Grammar Fix</h1>
        <p>Detect and correct grammatical errors with detailed corrections.</p>
      </div>

      <div className="card" style={{ maxWidth: 760 }}>
        <div className="card-header">
          <div className="card-icon" style={{ background: 'rgba(16,185,129,0.12)', color: 'var(--clr-success)' }}>
            <ShieldCheck size={20} />
          </div>
          <div>
            <div className="card-title">Grammar Checker</div>
            <div className="card-desc">Powered by a grammar correction transformer model</div>
          </div>
        </div>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <FormGroup label="Input Text">
            <textarea
              id="grammar-input"
              className="input"
              placeholder="Type or paste text to check…"
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={5}
            />
          </FormGroup>
          <div style={{ display: 'flex', gap: 10 }}>
            <button id="grammar-submit-btn" className="btn btn-primary" type="submit" disabled={loading || !text.trim()}>
              {loading ? <Spinner /> : <ArrowRight size={16} />}
              {loading ? 'Checking…' : 'Check Grammar'}
            </button>
            <button className="btn btn-secondary" type="button" onClick={() => { setText(''); setResult(null); }}>
              Clear
            </button>
          </div>
        </form>

        {result && (
          <div className="mt-4" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div className="form-label">Corrected Text</div>
            <ResultBox text={result.corrected_text} />
            {result.corrections?.length > 0 && (
              <div>
                <div className="form-label mb-2">
                  Corrections ({result.corrections.length})&nbsp;
                  <span className="badge badge-green">{result.corrections.length} found</span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {result.corrections.map((c, i) => (
                    <div key={i} className="card-glass" style={{ padding: '10px 14px' }}>
                      <span style={{ color: 'var(--clr-error)', textDecoration: 'line-through', marginRight: 8 }}>{c.original}</span>
                      <span style={{ color: 'var(--clr-success)' }}>→ {c.replacement}</span>
                      {c.description && <div className="text-xs text-muted mt-2">{c.description}</div>}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </PageWrapper>
  );
}


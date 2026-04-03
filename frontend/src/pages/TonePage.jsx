import { useState } from 'react';
import PageWrapper from '../components/PageWrapper';
import { Mic2, ArrowRight } from 'lucide-react';
import { tone } from '../api';
import { useToast } from '../context/ToastContext';
import { ResultBox, Spinner, FormGroup } from '../components/ui';

const TONES = ['professional', 'friendly', 'academic', 'assertive', 'empathetic', 'humorous'];

export default function TonePage() {
  const [text, setText] = useState('');
  const [targetTone, setTargetTone] = useState('professional');
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);
  const toast = useToast();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim()) return;
    setLoading(true);
    setResult('');
    try {
      const { data } = await tone(text, targetTone);
      setResult(data.toned_text);
      toast('Tone adjusted!', 'success');
    } catch (err) {
      toast(err.response?.data?.detail ?? 'Tone adjustment failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageWrapper>
      <div className="page-header">
        <h1>Tone Adjustment</h1>
        <p>Transform how your writing sounds — from academic to friendly and beyond.</p>
      </div>

      <div className="card" style={{ maxWidth: 760 }}>
        <div className="card-header">
          <div className="card-icon" style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--clr-warning)' }}>
            <Mic2 size={20} />
          </div>
          <div>
            <div className="card-title">Tone Adjuster</div>
            <div className="card-desc">Style transfer across 6 writing tones</div>
          </div>
        </div>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <FormGroup label="Input Text">
            <textarea
              id="tone-input"
              className="input"
              placeholder="Paste text to adjust tone…"
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={5}
            />
          </FormGroup>

          <FormGroup label="Target Tone">
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {TONES.map((t) => (
                <button
                  key={t}
                  type="button"
                  id={`tone-btn-${t}`}
                  className={`btn ${targetTone === t ? 'btn-primary' : 'btn-secondary'} btn-sm`}
                  onClick={() => setTargetTone(t)}
                >
                  {t.charAt(0).toUpperCase() + t.slice(1)}
                </button>
              ))}
            </div>
          </FormGroup>

          <div style={{ display: 'flex', gap: 10 }}>
            <button id="tone-submit-btn" className="btn btn-primary" type="submit" disabled={loading || !text.trim()}>
              {loading ? <Spinner /> : <ArrowRight size={16} />}
              {loading ? 'Adjusting…' : 'Adjust Tone'}
            </button>
            <button className="btn btn-secondary" type="button" onClick={() => { setText(''); setResult(''); }}>
              Clear
            </button>
          </div>
        </form>

        {result && (
          <div className="mt-4">
            <div className="form-label mb-2">Result — <span className="badge badge-amber">{targetTone}</span></div>
            <ResultBox text={result} />
          </div>
        )}
      </div>
    </PageWrapper>
  );
}


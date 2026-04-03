import { useState } from 'react';
import { Wand2, ArrowRight } from 'lucide-react';
import { paraphrase } from '../api';
import { useToast } from '../context/ToastContext';
import { ResultBox, Spinner, FormGroup } from '../components/ui';
import PageWrapper from '../components/PageWrapper';

const TONES = ['neutral', 'formal', 'casual', 'creative', 'concise'];

export default function ParaphrasePage() {
  const [text, setText] = useState('');
  const [tone, setTone] = useState('neutral');
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);
  const toast = useToast();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim()) return;
    setLoading(true);
    setResult('');
    try {
      const { data } = await paraphrase(text, tone);
      setResult(data.paraphrased_text);
      toast('Paraphrase complete!', 'success');
    } catch (err) {
      toast(err.response?.data?.detail ?? 'Paraphrase failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageWrapper>
      <div className="page-header">
        <h1>Paraphrase</h1>
        <p>Rewrite text while preserving its meaning, in any tone you choose.</p>
      </div>

      <div className="card" style={{ maxWidth: 760 }}>
        <div className="card-header">
          <div className="card-icon" style={{ background: 'rgba(124,58,237,0.15)', color: 'var(--clr-accent)' }}>
            <Wand2 size={20} />
          </div>
          <div>
            <div className="card-title">Paraphrase Text</div>
            <div className="card-desc">Tone-aware rewriting powered by a fine-tuned transformer</div>
          </div>
        </div>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <FormGroup label="Input Text">
            <textarea
              id="paraphrase-input"
              className="input"
              placeholder="Paste or type text to paraphrase…"
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={5}
            />
          </FormGroup>

          <FormGroup label="Target Tone">
            <select
              id="paraphrase-tone-select"
              className="input"
              value={tone}
              onChange={(e) => setTone(e.target.value)}
            >
              {TONES.map((t) => (
                <option key={t} value={t}>{t.charAt(0).toUpperCase() + t.slice(1)}</option>
              ))}
            </select>
          </FormGroup>

          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <button id="paraphrase-submit-btn" className="btn btn-primary" type="submit" disabled={loading || !text.trim()}>
              {loading ? <Spinner /> : <ArrowRight size={16} />}
              {loading ? 'Paraphrasing…' : 'Paraphrase'}
            </button>
            <button className="btn btn-secondary" type="button" onClick={() => { setText(''); setResult(''); }}>
              Clear
            </button>
          </div>
        </form>

        {result && (
          <div className="mt-4">
            <div className="form-label mb-2">Result</div>
            <ResultBox text={result} />
          </div>
        )}
      </div>
    </PageWrapper>
  );
}

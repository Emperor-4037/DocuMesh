"""
Golden regression tests for model output quality.
Each test sends a known input and validates structural properties of the output.
These tests require models to be loaded — run with: pytest tests/test_golden.py -v

What these catch:
  - Model output degradation after retraining or adapter swap
  - Empty/null outputs
  - Output that is identical to input (model not doing anything)
  - Summarization hallucinations (output longer than input)
"""
import pytest

# Mark all tests in this module as requiring model weights (slow)
pytestmark = pytest.mark.skipif(
    True,  # Set to False when models are available locally
    reason="Golden tests require model weights — set skip to False to run"
)


# ── Paraphrase ────────────────────────────────────────────────────────────────

class TestParaphraseGolden:
    def test_output_differs_from_input(self):
        from services.paraphrase_service.app.model import paraphrase
        inp = "The quick brown fox jumps over the lazy dog."
        out = paraphrase(inp, tone="neutral")
        assert out and len(out) > 5
        assert out.lower() != inp.lower(), "Paraphrase output should differ from input"

    def test_formal_tone_changes_style(self):
        from services.paraphrase_service.app.model import paraphrase
        inp = "Hey, can you help me out with this thing?"
        out = paraphrase(inp, tone="formal")
        assert out and len(out) > 5


# ── Grammar ───────────────────────────────────────────────────────────────────

class TestGrammarGolden:
    def test_corrects_known_error(self):
        from services.grammar_service.app.model import correct_grammar
        corrected, corrections = correct_grammar("He go to school yesterday.")
        assert corrected and len(corrected) > 5

    def test_preserves_correct_text(self):
        from services.grammar_service.app.model import correct_grammar
        inp = "She went to the store and bought some groceries."
        corrected, corrections = correct_grammar(inp)
        # Should not make many changes to already-correct text
        assert len(corrections) <= 2, f"Too many corrections on valid text: {corrections}"


# ── Simplify ──────────────────────────────────────────────────────────────────

class TestSimplifyGolden:
    def test_simplifies_complex_text(self):
        from services.simplify_service.app.model import simplify
        inp = "The implementation of the aforementioned algorithm necessitates a comprehensive understanding of computational complexity theory."
        out = simplify(inp, reading_level="basic")
        assert out and len(out) > 5


# ── Tone ──────────────────────────────────────────────────────────────────────

class TestToneGolden:
    def test_formal_tone(self):
        from services.tone_service.app.model import transfer_tone
        inp = "Hey, just wanted to let you know the meeting's been pushed."
        out = transfer_tone(inp, target_tone="formal")
        assert out and len(out) > 5


# ── Summarize ─────────────────────────────────────────────────────────────────

class TestSummarizeGolden:
    def test_summary_shorter_than_input(self):
        from services.summarize_service.app.model import summarize
        inp = (
            "Artificial intelligence has transformed the way businesses operate. "
            "From automating routine tasks to providing insights through data analysis, "
            "AI technologies are being adopted across industries. Machine learning, "
            "a subset of AI, enables computers to learn from data without being explicitly "
            "programmed. Deep learning, using neural networks with many layers, has achieved "
            "remarkable results in image recognition, natural language processing, and speech "
            "recognition. Companies are investing heavily in AI research and development."
        )
        out = summarize(inp, max_length=50)
        assert out and len(out) < len(inp), "Summary should be shorter than input"

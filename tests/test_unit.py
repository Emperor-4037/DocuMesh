"""
Unit tests for shared modules: schemas, config, model_config, model_utils.
Tests pure logic that doesn't require model weights.

What these catch:
  - Schema validation regressions (missing fields, wrong types)
  - Config default breakage
  - Model config wiring errors
  - Device detection logic bugs
"""
import pytest
from pydantic import ValidationError


# ── Schema Tests ──────────────────────────────────────────────────────────────

class TestSchemas:
    def test_paraphrase_request_valid(self):
        from shared.schemas import ParaphraseRequest
        r = ParaphraseRequest(text="Hello world", tone="formal")
        assert r.text == "Hello world"
        assert r.tone == "formal"

    def test_paraphrase_request_empty_text_rejected(self):
        from shared.schemas import ParaphraseRequest
        with pytest.raises(ValidationError):
            ParaphraseRequest(text="")

    def test_grammar_response_has_corrections_list(self):
        from shared.schemas import GrammarResponse
        r = GrammarResponse(corrected_text="Hello.", corrections=[])
        assert r.corrections == []
        assert r.success is True

    def test_correction_schema(self):
        from shared.schemas import Correction
        c = Correction(
            original="gos",
            replacement="goes",
            start_index=3,
            end_index=6,
            description="Replace: 'gos' → 'goes'"
        )
        assert c.original == "gos"

    def test_simplify_request_defaults(self):
        from shared.schemas import SimplifyRequest
        r = SimplifyRequest(text="Complex text")
        assert r.reading_level == "basic"

    def test_tone_request_defaults(self):
        from shared.schemas import ToneRequest
        r = ToneRequest(text="Some text")
        assert r.target_tone == "professional"

    def test_summarize_request_defaults(self):
        from shared.schemas import SummarizeRequest
        r = SummarizeRequest(text="Long text here")
        assert r.max_length == 150

    def test_rag_query_request(self):
        from shared.schemas import RAGQueryRequest
        r = RAGQueryRequest(query="What is AI?", top_k=3)
        assert r.top_k == 3

    def test_rag_response_with_sources(self):
        from shared.schemas import RAGQueryResponse, ChunkMetadata
        r = RAGQueryResponse(
            answer="AI is...",
            sources=[ChunkMetadata(source="doc.pdf", page=1)]
        )
        assert len(r.sources) == 1
        assert r.sources[0].source == "doc.pdf"

    def test_error_response(self):
        from shared.schemas import ErrorResponse
        r = ErrorResponse(message="Something failed")
        assert r.success is False


# ── Config Tests ──────────────────────────────────────────────────────────────

class TestConfig:
    def test_default_secret_key(self):
        from shared.config import settings
        assert settings.SECRET_KEY  # must not be empty

    def test_sqlalchemy_uri(self):
        from shared.config import settings
        uri = settings.SQLALCHEMY_DATABASE_URI
        assert uri.startswith("postgresql://")

    def test_redis_defaults(self):
        from shared.config import settings
        assert settings.REDIS_PORT == 6379


# ── Model Config Tests ────────────────────────────────────────────────────────

class TestModelConfig:
    def test_paraphrase_config_defaults(self):
        from shared.model_config import PARAPHRASE_CONFIG
        assert "t5" in PARAPHRASE_CONFIG.base_model.lower() or "paraphras" in PARAPHRASE_CONFIG.base_model.lower()
        assert PARAPHRASE_CONFIG.model_type == "seq2seq"

    def test_grammar_config_defaults(self):
        from shared.model_config import GRAMMAR_CONFIG
        assert GRAMMAR_CONFIG.model_type == "seq2seq"
        assert GRAMMAR_CONFIG.num_beams >= 1

    def test_rag_llm_config_is_causal(self):
        from shared.model_config import RAG_LLM_CONFIG
        assert RAG_LLM_CONFIG.model_type == "causal"

    def test_config_is_frozen(self):
        from shared.model_config import PARAPHRASE_CONFIG
        with pytest.raises(Exception):  # FrozenInstanceError
            PARAPHRASE_CONFIG.base_model = "other-model"

    def test_all_configs_have_base_model(self):
        from shared.model_config import (
            PARAPHRASE_CONFIG, GRAMMAR_CONFIG, SIMPLIFY_CONFIG,
            TONE_CONFIG, SUMMARIZE_CONFIG, RAG_LLM_CONFIG,
        )
        for cfg in [PARAPHRASE_CONFIG, GRAMMAR_CONFIG, SIMPLIFY_CONFIG,
                     TONE_CONFIG, SUMMARIZE_CONFIG, RAG_LLM_CONFIG]:
            assert cfg.base_model, f"Missing base_model in {cfg}"


# ── Model Utils Tests ─────────────────────────────────────────────────────────

class TestModelUtils:
    def test_device_is_valid(self):
        from shared.model_utils import DEVICE
        assert DEVICE in ("cuda", "mps", "cpu")

    def test_dtype_matches_device(self):
        from shared.model_utils import DEVICE, DTYPE
        import torch
        if DEVICE == "cuda":
            assert DTYPE == torch.float16
        else:
            assert DTYPE == torch.float32

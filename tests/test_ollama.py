import json

from adaptq.runtime_py.backends import ollama as ollama_backend
from adaptq.runtime_py.metadata import ModelConfig


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        chunks = [
            {"response": "hello", "done": False},
            {
                "response": " world",
                "done": True,
                "prompt_eval_count": 4,
                "eval_count": 2,
            },
        ]
        return [json.dumps(chunk).encode("utf-8") for chunk in chunks]


class _FakeRequests:
    def post(self, *args, **kwargs):
        return _FakeResponse()


def test_ollama_generate_does_not_fabricate_kv_stats(monkeypatch):
    adapter = ollama_backend.OllamaAdapter()
    adapter._model_name = "test-model"
    adapter._model_cfg = ModelConfig(model_path="test-model")
    monkeypatch.setattr(ollama_backend, "requests", _FakeRequests())

    result = adapter.generate("hello", max_new_tokens=2)

    assert result.success
    assert result.text == "hello world"
    assert result.n_prompt_tokens == 4
    assert result.n_generated_tokens == 2
    assert result.kv_stats.kv_bytes_fp16 == 0
    assert result.kv_stats.kv_bytes_adaptq == 0
    assert result.kv_stats.n_tokens_cached == 0
    assert result.kv_stats.compression_ratio == 0.0
    assert "KV cache:" not in result.summary()


def test_ollama_get_kv_stats_returns_unavailable_defaults(monkeypatch):
    adapter = ollama_backend.OllamaAdapter()
    adapter._model_name = "test-model"
    adapter._model_cfg = ModelConfig(model_path="test-model")
    monkeypatch.setattr(ollama_backend, "requests", _FakeRequests())

    adapter.generate("hello", max_new_tokens=2)
    stats = adapter.get_kv_stats()

    assert stats.kv_bytes_fp16 == 0
    assert stats.kv_bytes_adaptq == 0
    assert stats.n_tokens_cached == 0
    assert stats.compression_ratio == 0.0

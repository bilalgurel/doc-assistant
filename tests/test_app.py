"""Tests run offline: a small fake embedder replaces the real model, and the LLM call is mocked."""
import hashlib
import json
import math
import re

import pytest
from chromadb import Documents, EmbeddingFunction, Embeddings
from fastapi.testclient import TestClient

from app import config, ingest, llm, main
from app.store import Store


class FakeEmbedder(EmbeddingFunction):
    """Bag-of-words hashing embedder. Good enough to test retrieval logic offline."""

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        out = []
        for text in input:
            vec = [0.0] * 256
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out

    @staticmethod
    def name() -> str:
        return "fake-hash"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return FakeEmbedder()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "LLM_MODEL", "test-model")
    monkeypatch.setattr(config, "ADMIN_TOKEN", "secret")
    monkeypatch.setattr(config, "MAX_DISTANCE", 0.8)
    monkeypatch.setattr(config, "RATE_LIMIT_PER_MINUTE", 50)
    store = Store(tmp_path / "chroma", FakeEmbedder())
    monkeypatch.setattr(main, "_store", store)
    main._rate_hits.clear()

    doc_a = tmp_path / "refunds.md"
    doc_a.write_text("Cancellations made 60 days or more before departure receive a full refund minus the deposit.")
    doc_b = tmp_path / "secret.md"
    doc_b.write_text("The internal margin on premium trips is forty percent and must stay confidential.")
    ingest.ingest(store, "alice", [str(doc_a)])
    ingest.ingest(store, "bob", [str(doc_b)])
    return TestClient(main.app)


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        event = re.search(r"^event: (.+)$", block, re.M).group(1)
        data = json.loads(re.search(r"^data: (.+)$", block, re.M).group(1))
        events.append((event, data))
    return events


def fake_llm(answer):
    def _stream(system, user, model, max_tokens):
        _stream.calls.append({"system": system, "user": user, "model": model})
        for word in answer.split(" "):
            yield "text", word + " "
        yield "usage", {"input_tokens": 120, "output_tokens": 30}

    _stream.calls = []
    return _stream


def test_chunking_overlaps_and_covers_text():
    text = "Sentence number one. " * 200
    chunks = ingest.chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    assert chunks[-1].endswith("one.")


def test_reingest_replaces_instead_of_duplicating(tmp_path):
    store = Store(tmp_path / "c", FakeEmbedder())
    doc = tmp_path / "a.md"
    doc.write_text("First version of the policy.")
    ingest.ingest(store, "x", [str(doc)])
    doc.write_text("Second version of the policy.")
    ingest.ingest(store, "x", [str(doc)])
    assert store.col.count() == 1
    assert "Second" in store.search("x", "policy", 3)[0]["text"]


def test_answer_streams_with_cited_sources_and_is_logged(client, monkeypatch):
    stream = fake_llm("You get a full refund minus the deposit [1].")
    monkeypatch.setattr(llm, "stream_answer", stream)
    monkeypatch.setattr(config, "PRICE_INPUT_PER_MTOK", 1.0)
    monkeypatch.setattr(config, "PRICE_OUTPUT_PER_MTOK", 5.0)

    res = client.post("/chat", json={"customer_id": "alice", "question": "refund if I cancel 60 days before departure?"})
    assert res.status_code == 200
    events = parse_sse(res.text)
    answer = "".join(d["text"] for e, d in events if e == "token")
    assert "full refund" in answer
    sources = next(d for e, d in events if e == "sources")["sources"]
    assert sources and sources[0]["source"] == "refunds.md"
    done = next(d for e, d in events if e == "done")
    assert done["refused"] is False
    assert done["cost_usd"] == pytest.approx(120 / 1e6 * 1.0 + 30 / 1e6 * 5.0)

    # The model only ever saw alice's document.
    assert "confidential" not in stream.calls[0]["user"]
    assert stream.calls[0]["model"] == "test-model"

    log = client.get("/admin/api/log", headers={"X-Admin-Token": "secret"}).json()["rows"]
    assert log[0]["question"].startswith("refund") and log[0]["input_tokens"] == 120


def test_customers_cannot_read_each_others_documents(client, monkeypatch):
    stream = fake_llm("should not be called")
    monkeypatch.setattr(llm, "stream_answer", stream)
    res = client.post("/chat", json={"customer_id": "alice", "question": "internal margin premium trips confidential"})
    events = parse_sse(res.text)
    for call in stream.calls:
        assert "forty percent" not in call["user"]
    if not stream.calls:
        assert next(d for e, d in events if e == "done")["refused"] is True


def test_out_of_scope_question_is_refused_without_calling_model(client, monkeypatch):
    stream = fake_llm("should not be called")
    monkeypatch.setattr(llm, "stream_answer", stream)
    res = client.post("/chat", json={"customer_id": "alice", "question": "quantum chromodynamics lattice"})
    events = parse_sse(res.text)
    assert "".join(d["text"] for e, d in events if e == "token") == config.REFUSAL_MESSAGE
    assert next(d for e, d in events if e == "done")["refused"] is True
    assert stream.calls == []


def test_model_refusal_returns_no_sources(client, monkeypatch):
    monkeypatch.setattr(llm, "stream_answer", fake_llm(config.REFUSAL_MESSAGE))
    res = client.post("/chat", json={"customer_id": "alice", "question": "refund deposit departure cancellations"})
    events = parse_sse(res.text)
    assert next(d for e, d in events if e == "sources")["sources"] == []
    assert next(d for e, d in events if e == "done")["refused"] is True


def test_rate_limit(client, monkeypatch):
    monkeypatch.setattr(config, "RATE_LIMIT_PER_MINUTE", 2)
    monkeypatch.setattr(llm, "stream_answer", fake_llm("ok [1]"))
    body = {"customer_id": "alice", "question": "refund"}
    assert client.post("/chat", json=body).status_code == 200
    assert client.post("/chat", json=body).status_code == 200
    assert client.post("/chat", json=body).status_code == 429


def test_admin_requires_token_and_can_edit_prompt(client):
    assert client.get("/admin/api/prompt").status_code == 401
    assert client.get("/admin/api/prompt", headers={"X-Admin-Token": "wrong"}).status_code == 401
    headers = {"X-Admin-Token": "secret"}
    res = client.put("/admin/api/prompt", json={"prompt": "Custom prompt. {refusal}"}, headers=headers)
    assert res.status_code == 200
    assert client.get("/admin/api/prompt", headers=headers).json()["prompt"].startswith("Custom prompt")
    assert main.get_system_prompt() == f"Custom prompt. {config.REFUSAL_MESSAGE}"


def test_gemini_adapter_streams_text_and_counts_thinking_tokens(monkeypatch):
    from types import SimpleNamespace

    from google import genai

    seen = {}

    class FakeModels:
        def generate_content_stream(self, model, contents, config):
            seen.update(model=model, contents=contents, system=config.system_instruction,
                        max_tokens=config.max_output_tokens)
            yield SimpleNamespace(text=None, usage_metadata=None)  # thinking-only chunk
            yield SimpleNamespace(text="Hello ", usage_metadata=None)
            yield SimpleNamespace(
                text="world [1]",
                usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=7, thoughts_token_count=20),
            )

    monkeypatch.setattr(genai, "Client", lambda *a, **k: SimpleNamespace(models=FakeModels()))
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    events = list(llm.stream_answer("SYS", "USER", "some-model", 321))
    assert events == [
        ("text", "Hello "),
        ("text", "world [1]"),
        ("usage", {"input_tokens": 100, "output_tokens": 27}),
    ]
    assert seen == {"model": "some-model", "contents": "USER", "system": "SYS", "max_tokens": 321}


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "nope")
    with pytest.raises(ValueError):
        list(llm.stream_answer("s", "u", "m", 10))

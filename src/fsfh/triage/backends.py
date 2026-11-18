"""Two LLM backends behind one interface, plus a disk cache.

* ``AnthropicBackend``: Claude via the official SDK with structured output
  (``messages.parse`` validates the reply against the pydantic schema).
* ``OllamaBackend``: any OpenAI-compatible ``/v1/chat/completions`` endpoint
  (Ollama by default), JSON mode with the schema in the prompt. Plain HTTP.

``CachedBackend`` wraps either: the request (backend id, model, system, user,
schema) is hashed; a hit is served from ``<cache_dir>/<hash>.json``. With
``offline=True`` a miss raises instead of calling the network, which is how
tests and ``make results`` run.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from fsfh.triage.schema import TriageReport

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_OLLAMA_MODEL = "llama3.1"
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"


class Backend(Protocol):
    id: str
    model: str

    def complete(self, system: str, user: str) -> dict:
        """Return the raw JSON object the model produced (validated by the caller)."""


class AnthropicBackend:
    id = "anthropic"

    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL, api_key: str | None = None) -> None:
        import anthropic
        from dotenv import load_dotenv

        load_dotenv()  # ANTHROPIC_API_KEY from .env, never committed
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, system: str, user: str) -> dict:
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=TriageReport,
            output_config={"effort": "low"},
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"model refused: {response.stop_details}")
        return response.parsed_output.model_dump()


class OllamaBackend:
    id = "ollama"

    def __init__(
        self, model: str = DEFAULT_OLLAMA_MODEL, base_url: str = DEFAULT_OLLAMA_URL
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete(self, system: str, user: str) -> dict:
        schema = TriageReport.model_json_schema()
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": f"{system}\n\nReply with JSON matching: {json.dumps(schema)}",
                },
                {"role": "user", "content": user},
            ],
            # Constrained decoding: small local models echo or drop fields in
            # free-form JSON mode; json_schema makes validity a decoder property.
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "triage_report", "schema": schema},
            },
            "temperature": 0,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.load(resp)
        return json.loads(payload["choices"][0]["message"]["content"])


class CachedBackend:
    """Disk cache in front of a backend. Keyed on everything that changes the answer."""

    def __init__(self, inner: Backend | None, cache_dir: Path, offline: bool = False) -> None:
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.offline = offline
        self.hits = 0
        self.misses = 0
        self.id = inner.id if inner else "cache"
        self.model = inner.model if inner else "cache"

    def key(self, system: str, user: str) -> str:
        payload = json.dumps(
            {"backend": self.id, "model": self.model, "system": system, "user": user,
             "schema": TriageReport.model_json_schema()},
            sort_keys=True,
        )  # fmt: skip
        return hashlib.sha256(payload.encode()).hexdigest()[:24]

    def complete(self, system: str, user: str) -> dict:
        path = self.cache_dir / f"{self.key(system, user)}.json"
        if path.is_file():
            self.hits += 1
            return json.loads(path.read_text())["response"]
        if self.offline or self.inner is None:
            raise FileNotFoundError(f"triage cache miss in offline mode: {path.name}")
        self.misses += 1
        response = self.inner.complete(system, user)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"backend": self.id, "model": self.model, "user": user, "response": response},
                indent=2,
            )
            + "\n"
        )
        return response


def make_backend(name: str, model: str | None, cache_dir: Path, offline: bool) -> CachedBackend:
    if offline:
        inner: Backend | None = None
        cached = CachedBackend(inner, cache_dir, offline=True)
        cached.id, cached.model = name, model or _default_model(name)
        return cached
    if name == "anthropic":
        inner = AnthropicBackend(model or DEFAULT_ANTHROPIC_MODEL)
    elif name == "ollama":
        inner = OllamaBackend(model or DEFAULT_OLLAMA_MODEL)
    else:
        raise ValueError(f"unknown backend {name!r}; use 'anthropic' or 'ollama'")
    return CachedBackend(inner, cache_dir)


def _default_model(name: str) -> str:
    return DEFAULT_ANTHROPIC_MODEL if name == "anthropic" else DEFAULT_OLLAMA_MODEL


def validate(raw: dict) -> TriageReport:
    try:
        return TriageReport.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"triage report failed validation: {e}") from e

"""Phase 4 tests. No network: a fake backend for the logic, the committed cache for the agent."""

from __future__ import annotations

from pathlib import Path

import pytest

from fsfh.runlog import read_run
from fsfh.triage.agent import SYSTEM_PROMPT, triage_run
from fsfh.triage.backends import CachedBackend, make_backend, validate

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "runs" / "fixtures"
MODELS = ROOT / "models"
CACHE = ROOT / "tests" / "fixtures" / "triage_cache"


class FakeBackend:
    id, model = "fake", "fake-1"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str) -> dict:
        self.calls += 1
        return {"fault_type": "stuck_airspeed", "confidence": 0.9,
                "evidence": ["staleness alarm"], "recommended_check": "inspect pitot"}  # fmt: skip


def test_schema_rejects_bad_reports() -> None:
    with pytest.raises(ValueError):
        validate({"fault_type": "gremlins", "confidence": 0.5, "evidence": ["x"],
                  "recommended_check": "y"})  # fmt: skip
    with pytest.raises(ValueError):
        validate({"fault_type": "unknown", "confidence": 1.5, "evidence": ["x"],
                  "recommended_check": "y"})  # fmt: skip
    with pytest.raises(ValueError):
        validate({"fault_type": "unknown", "confidence": 0.5, "evidence": [],
                  "recommended_check": "y"})  # fmt: skip


def test_cache_serves_second_call_from_disk(tmp_path: Path) -> None:
    fake = FakeBackend()
    cached = CachedBackend(fake, tmp_path)
    assert cached.complete(SYSTEM_PROMPT, "window A") == cached.complete(SYSTEM_PROMPT, "window A")
    assert fake.calls == 1 and cached.hits == 1
    offline = CachedBackend(None, tmp_path, offline=True)
    offline.id, offline.model = fake.id, fake.model
    assert offline.complete(SYSTEM_PROMPT, "window A")["fault_type"] == "stuck_airspeed"
    with pytest.raises(FileNotFoundError):
        offline.complete(SYSTEM_PROMPT, "window B")


@pytest.mark.skipif(not CACHE.is_dir(), reason="no cached triage responses committed yet")
def test_agent_on_fixture_from_cache() -> None:
    """Uses only committed responses under tests/fixtures/triage_cache (never the network)."""
    backend = make_backend("anthropic", None, CACHE, offline=True)
    fixture = next(p for p in sorted(FIXTURES.iterdir()) if "stuck_airspeed" in p.name)
    run = read_run(fixture)
    results = triage_run(run.sensors, MODELS, backend)
    assert results, "no flagged window on the stuck_airspeed fixture"
    t0, t1, report = results[0]
    assert report.fault_type == "stuck_airspeed"
    assert 0.0 <= report.confidence <= 1.0 and report.evidence

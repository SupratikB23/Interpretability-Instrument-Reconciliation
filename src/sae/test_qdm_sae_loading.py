"""Regression tests for revision-pinned SAE loading across sae_lens versions.

sae_lens through 6.49.x has no ``revision`` argument on ``SAE.from_pretrained``;
the reproduction pin is enforced instead by verifying the release repo HEAD
still equals the pinned sha (byte-identical download) and refusing on drift.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import ClassVar

import huggingface_hub
import pytest

from src.sae import qdm


class _CapturingSAE:
    """Fake SAE whose from_pretrained has NO revision parameter (sae_lens 6.49)."""

    calls: ClassVar[list[dict]] = []
    result: ClassVar[object] = "bare-sae"

    @classmethod
    def from_pretrained(cls, release, sae_id, device="cpu", dtype="float32"):
        cls.calls.append({"release": release, "sae_id": sae_id, "device": device})
        return cls.result


class _RevisionAwareSAE:
    """Fake SAE whose from_pretrained exposes revision (future sae_lens)."""

    calls: ClassVar[list[dict]] = []

    @classmethod
    def from_pretrained(cls, release, sae_id, device="cpu", revision=None):
        cls.calls.append({"release": release, "sae_id": sae_id, "revision": revision})
        return "revision-loaded-sae"


def _fake_head(monkeypatch, sha: str) -> list[str]:
    looked_up: list[str] = []

    class FakeHfApi:
        def model_info(self, repo_id, timeout=None):
            looked_up.append(repo_id)
            return SimpleNamespace(sha=sha)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    return looked_up


def test_drift_check_network_failure_refuses_loudly(monkeypatch):
    """A blocked huggingface.co must fail fast with guidance, never hang silently."""
    monkeypatch.setattr(qdm, "_sae_release_repo_id", lambda release: "google/gemma-scope-2b-pt-res")

    class BrokenHfApi:
        def model_info(self, repo_id, timeout=None):
            raise TimeoutError("endpoint unreachable")

    monkeypatch.setattr(huggingface_hub, "HfApi", BrokenHfApi)
    with pytest.raises(RuntimeError, match="cannot verify SAE checkpoint pin"):
        qdm._assert_sae_checkpoint_not_drifted(
            "gemma-scope-2b-pt-res", "fd571b47c1c64851e9b1989792367b9babb4af63"
        )


@pytest.fixture(autouse=True)
def _clean_state():
    _CapturingSAE.calls = []
    _RevisionAwareSAE.calls = []
    yield


def test_pinned_revision_loads_when_repo_head_matches(monkeypatch):
    pinned = "fd571b47c1c64851e9b1989792367b9babb4af63"
    monkeypatch.setitem(sys.modules, "sae_lens", SimpleNamespace(SAE=_CapturingSAE))
    monkeypatch.setattr(qdm, "_sae_release_repo_id", lambda release: "google/gemma-scope-2b-pt-res")
    looked_up = _fake_head(monkeypatch, pinned)

    loaded = qdm._load_sae(
        release="gemma-scope-2b-pt-res", sae_id="layer_12/width_16k/average_l0_82",
        revision=pinned, device="cpu",
    )

    assert loaded == "bare-sae"
    assert looked_up == ["google/gemma-scope-2b-pt-res"]
    assert _CapturingSAE.calls == [
        {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_16k/average_l0_82", "device": "cpu"}
    ], "revision must not be passed to a sae_lens that does not support it"


def test_drifted_repo_head_refuses_to_load(monkeypatch):
    monkeypatch.setitem(sys.modules, "sae_lens", SimpleNamespace(SAE=_CapturingSAE))
    monkeypatch.setattr(qdm, "_sae_release_repo_id", lambda release: "google/gemma-scope-2b-pt-res")
    _fake_head(monkeypatch, "deadbeef" * 5)

    with pytest.raises(RuntimeError, match="drift detected"):
        qdm._load_sae(
            release="gemma-scope-2b-pt-res", sae_id="layer_12/width_16k/average_l0_82",
            revision="fd571b47c1c64851e9b1989792367b9babb4af63", device="cpu",
        )
    assert _CapturingSAE.calls == [], "no SAE may be loaded from a drifted checkpoint"


def test_revision_passthrough_when_sae_lens_supports_it(monkeypatch):
    pinned = "fd571b47c1c64851e9b1989792367b9babb4af63"
    monkeypatch.setitem(sys.modules, "sae_lens", SimpleNamespace(SAE=_RevisionAwareSAE))
    spy = _fake_head(monkeypatch, "should-not-be-used")

    loaded = qdm._load_sae(
        release="gemma-scope-2b-pt-res", sae_id="layer_12/width_16k/average_l0_82",
        revision=pinned, device="cpu",
    )

    assert loaded == "revision-loaded-sae"
    assert _RevisionAwareSAE.calls[0]["revision"] == pinned
    assert spy == [], "no drift check is needed when the pin can be passed through"


def test_tuple_return_from_older_sae_lens_is_unwrapped(monkeypatch):
    monkeypatch.setitem(sys.modules, "sae_lens", SimpleNamespace(SAE=_CapturingSAE))
    _CapturingSAE.result = ("the-sae", {"cfg": True})

    loaded = qdm._load_sae(
        release="gemma-scope-2b-pt-res", sae_id="layer_12/width_16k/average_l0_82",
        revision=None, device="cpu",
    )
    assert loaded == "the-sae"


def test_unknown_release_refuses_pin_enforcement(monkeypatch):
    monkeypatch.setattr(qdm, "_sae_release_repo_id", lambda release: None)
    with pytest.raises(RuntimeError, match="no HF repo known"):
        qdm._assert_sae_checkpoint_not_drifted("not-a-real-release", "abc123")

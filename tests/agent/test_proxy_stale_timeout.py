"""Tests for loopback-proxy vs real-local-engine stale-stream gating.

The stream stale-timeout (and the matching httpx read-timeout) is disabled
only for a *detected* local inference engine. A loopback proxy that merely
fronts a remote backend — e.g. the cursor-openai-api proxy on 127.0.0.1:8080,
which exposes none of the engine-specific probe endpoints — is loopback by
address but must keep a finite, context-scaled stale timeout so a slow-but-
healthy remote call stays distinguishable from a hang.

These pin:
- detect_local_server_type_cached memoizes the (expensive) probe per base_url
- the production gate (real is_local_endpoint + cached probe) disables the
  detector only for a real engine, never for a bare loopback proxy
"""

import pytest
from unittest.mock import patch

import agent.model_metadata as model_metadata
from agent.model_metadata import detect_local_server_type_cached, is_local_endpoint


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    """Reset the memoized probe results around each test."""
    model_metadata._local_server_type_cache.clear()
    yield
    model_metadata._local_server_type_cache.clear()


def _stale_detector_disabled(base_url, base_timeout=180.0):
    """Mirror the production gate in chat_completion_helpers (both the stale
    and read-timeout branches share this exact condition)."""
    return (
        base_timeout == 180.0
        and bool(base_url)
        and is_local_endpoint(base_url)
        and detect_local_server_type_cached(base_url) is not None
    )


class TestDetectLocalServerTypeCached:
    """The cached wrapper must memoize so the hot path never re-probes."""

    def test_caches_per_base_url(self):
        calls = []

        def _fake_probe(base_url, api_key=""):
            calls.append(base_url)
            return "ollama"

        with patch.object(model_metadata, "detect_local_server_type", _fake_probe):
            first = detect_local_server_type_cached("http://127.0.0.1:11434/v1")
            second = detect_local_server_type_cached("http://127.0.0.1:11434/v1")

        assert first == "ollama"
        assert second == "ollama"
        assert len(calls) == 1  # probed once, served from cache thereafter

    def test_caches_none_results(self):
        """A bare proxy probes to None; that negative is cached too, otherwise
        the proxy would re-probe on every single stream request."""
        calls = []

        def _fake_probe(base_url, api_key=""):
            calls.append(base_url)
            return None

        with patch.object(model_metadata, "detect_local_server_type", _fake_probe):
            assert detect_local_server_type_cached("http://127.0.0.1:8080/v1") is None
            assert detect_local_server_type_cached("http://127.0.0.1:8080/v1") is None

        assert len(calls) == 1


class TestStaleDetectorGate:
    """The detector is disabled only for a real detected local engine."""

    def test_loopback_proxy_keeps_finite_timeout(self):
        """cursor-openai-api proxy: loopback address but probes to None."""
        with patch.object(model_metadata, "detect_local_server_type", lambda u, api_key="": None):
            assert is_local_endpoint("http://127.0.0.1:8080/v1") is True
            assert _stale_detector_disabled("http://127.0.0.1:8080/v1") is False

    def test_real_local_engine_disables_detector(self):
        """A detected Ollama engine keeps the long-prefill behavior (disabled)."""
        with patch.object(model_metadata, "detect_local_server_type", lambda u, api_key="": "ollama"):
            assert _stale_detector_disabled("http://127.0.0.1:11434/v1") is True

    def test_remote_endpoint_never_probed(self):
        """Remote cloud URLs short-circuit on is_local_endpoint, so the probe
        (which would error against a cloud host) is never reached."""
        def _boom(base_url, api_key=""):
            raise AssertionError("detect_local_server_type must not run for remote URLs")

        with patch.object(model_metadata, "detect_local_server_type", _boom):
            assert _stale_detector_disabled("https://api.openai.com/v1") is False

    def test_explicit_override_keeps_detector_finite(self):
        """A non-default base timeout (user set HERMES_STREAM_STALE_TIMEOUT)
        never disables the detector, even for a real engine."""
        with patch.object(model_metadata, "detect_local_server_type", lambda u, api_key="": "ollama"):
            assert _stale_detector_disabled("http://127.0.0.1:11434/v1", base_timeout=240.0) is False


class TestRealProductionGate:
    """Exercise the ACTUAL shipping predicate, not a hand-mirrored copy.

    Both the stale-stream and httpx read-timeout gates in
    chat_completion_helpers now route through the shared
    ``_is_detected_local_engine`` helper, so pinning that helper directly pins
    the production behavior: a loopback proxy stays finite (returns False), a
    detected local engine disables the detector (returns True), and remote URLs
    are never probed.
    """

    def test_loopback_proxy_returns_false(self):
        """cursor-openai-api proxy: loopback address, probes to None -> finite."""
        from agent.chat_completion_helpers import _is_detected_local_engine

        with patch.object(model_metadata, "detect_local_server_type", lambda u, api_key="": None):
            assert _is_detected_local_engine("http://127.0.0.1:8080/v1") is False

    def test_real_local_engine_returns_true(self):
        """A detected Ollama engine -> long-prefill behavior (detector off)."""
        from agent.chat_completion_helpers import _is_detected_local_engine

        with patch.object(model_metadata, "detect_local_server_type", lambda u, api_key="": "ollama"):
            assert _is_detected_local_engine("http://127.0.0.1:11434/v1") is True

    def test_remote_endpoint_never_probed(self):
        """Remote cloud URLs short-circuit on is_local_endpoint, so the probe
        (which would error against a cloud host) is never reached."""
        from agent.chat_completion_helpers import _is_detected_local_engine

        def _boom(base_url, api_key=""):
            raise AssertionError("detect_local_server_type must not run for remote URLs")

        with patch.object(model_metadata, "detect_local_server_type", _boom):
            assert _is_detected_local_engine("https://api.openai.com/v1") is False

    def test_api_key_forwarded_to_probe(self):
        """The live credentials are forwarded so an auth-gated local engine
        probes with parity to the real stream connection."""
        from agent.chat_completion_helpers import _is_detected_local_engine

        seen = {}

        def _probe(base_url, api_key=""):
            seen["api_key"] = api_key
            return "vllm"

        with patch.object(model_metadata, "detect_local_server_type", _probe):
            assert _is_detected_local_engine("http://127.0.0.1:8000/v1", "secret-key") is True
        assert seen["api_key"] == "secret-key"

"""Custom / Ollama (local) provider profile.

Covers any endpoint registered as provider="custom", including local
Ollama instances. Key quirks:
  - ollama_num_ctx → extra_body.options.num_ctx (local context window)
  - reasoning_config disabled → extra_body.think = False
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


def _current_profile_name() -> str | None:
    """Best-effort: resolve the profile this process is bound to from HERMES_HOME.

    Lets a fully-delegated run (empty Hermes toolset → Cursor SDK with its own
    tools) record which profile it originated from in the request metadata, so
    delegated sessions stay attributable even when a bare invocation falls back
    to the default home (#18594).
    """
    try:
        from hermes_constants import get_default_hermes_root, get_hermes_home

        home = get_hermes_home().resolve()
        root = get_default_hermes_root().resolve()
        if home == root:
            return "default"
        return home.name or None
    except Exception:
        return None


class CustomProfile(ProviderProfile):
    """Custom/Ollama local provider — think=false and num_ctx support."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        ollama_num_ctx: int | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}

        # Ollama context window
        if ollama_num_ctx:
            options = extra_body.get("options", {})
            options["num_ctx"] = ollama_num_ctx
            extra_body["options"] = options

        # Disable thinking when reasoning is turned off
        if reasoning_config and isinstance(reasoning_config, dict):
            _effort = (reasoning_config.get("effort") or "").strip().lower()
            _enabled = reasoning_config.get("enabled", True)
            if _effort == "none" or _enabled is False:
                extra_body["think"] = False

        top_level: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        session_id = (ctx.get("session_id") or "").strip()
        if session_id:
            # cursor-openai-api maps metadata.hermes_session_id → stable agent key
            metadata["hermes_session_id"] = session_id
        # Tag the originating profile so fully-delegated Cursor runs are
        # attributable to the profile that launched them.
        profile = _current_profile_name()
        if profile:
            metadata["hermes_profile"] = profile
        if metadata:
            top_level["metadata"] = metadata

        return extra_body, top_level

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Custom/Ollama: base_url is user-configured; fetch if set."""
        if not (base_url or self.base_url):
            return None
        return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)


custom = CustomProfile(
    name="custom",
    aliases=(
        "ollama",
        "local",
        "vllm",
        "llamacpp",
        "llama.cpp",
        "llama-cpp",
    ),
    env_vars=(),  # No fixed key — custom endpoint
    base_url="",  # User-configured
    # Without this, no max_tokens is sent and Ollama falls back to its internal
    # num_predict=128, truncating responses after a few tokens (#39281). This is
    # only a floor used when the user hasn't set model.max_tokens — they can
    # override per-model — so we set it generously rather than lowballing it.
    default_max_tokens=65536,
)

register_provider(custom)

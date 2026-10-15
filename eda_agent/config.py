from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any

PROVIDERS = ("openai", "anthropic", "gemini")

# Environment variables (or .env entries) are read per provider: <PREFIX>_API_KEY, <PREFIX>_MODEL,
# <PREFIX>_BASE_URL and <PREFIX>_EXTRA_BODY. LLM_PROVIDER selects the provider.
_ENV_PREFIX = {"openai": "OPENAI", "anthropic": "ANTHROPIC", "gemini": "GEMINI"}

DEFAULT_MODELS = {
    "openai": "gpt-5.2",
    "anthropic": "claude-opus-5-5",
    "gemini": "gemini-4-argon",
}

# Used when the caller leaves a setting unset (None).
_PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    # Same request settings as before multi-provider support.
    "openai": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 4096},
    # Current Claude models reject temperature/top_p, and adaptive thinking counts toward max_tokens.
    "anthropic": {"temperature": None, "top_p": None, "max_tokens": 16000},
    # Leave sampling and the output cap to the Gemini API defaults.
    "gemini": {"temperature": None, "top_p": None, "max_tokens": None},
}

_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "openai"
    model: str = DEFAULT_MODELS["openai"]
    api_key: str | None = None
    base_url: str | None = None
    organization: str | None = None  # OpenAI only
    reasoning_effort: str | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    # Anthropic only: server-side refusal fallbacks (set ANTHROPIC_FALLBACKS=off to disable).
    refusal_fallbacks: bool = True

    @property
    def has_credentials(self) -> bool:
        # The Anthropic SDK can also authenticate via ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile.
        return bool(self.api_key) or self.provider == "anthropic"

    def missing_credentials_message(self) -> str | None:
        if self.has_credentials:
            return None
        var = "GEMINI_API_KEY (or GOOGLE_API_KEY)" if self.provider == "gemini" else f"{_ENV_PREFIX[self.provider]}_API_KEY"
        return f"Missing {var} for provider '{self.provider}' (set it in .env or your shell, or pass --api-key)."


# Backwards-compatible name from when only OpenAI-compatible endpoints were supported.
OpenAIConfig = LLMConfig


def load_dotenv_file(env_file: str | os.PathLike[str] | None = None) -> Path | None:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Without `env_file`, looks for `.env` in the current directory and its parents, then in the
    repository root. Variables already set in the environment win, and empty values are skipped
    so that blank template entries (e.g. `OPENAI_BASE_URL=`) don't override SDK defaults.
    """
    from dotenv import dotenv_values, find_dotenv

    if env_file is not None:
        path = Path(env_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"env file not found: {path}")
    else:
        found = find_dotenv(usecwd=True)
        path = Path(found) if found else _REPO_ROOT / ".env"
        if not path.is_file():
            return None

    for key, value in dotenv_values(path).items():
        if value and key not in os.environ:
            os.environ[key] = value
    return path


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _parse_extra_body(value: dict[str, Any] | str | None, *, source: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source} must be valid JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{source} must be a JSON object.")
    return value


def load_llm_config(
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    organization: str | None = None,
    reasoning_effort: str | None = None,
    stream: bool | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    max_completion_tokens: int | None = None,
    extra_body: dict[str, Any] | str | None = None,
    env_file: str | os.PathLike[str] | None = None,
) -> LLMConfig:
    """Resolve the LLM endpoint: arguments > environment variables > .env file > defaults."""
    load_dotenv_file(env_file)

    provider = (provider or _env("LLM_PROVIDER") or "openai").strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider {provider!r}; expected one of: {', '.join(PROVIDERS)}.")
    prefix = _ENV_PREFIX[provider]

    env_key = _env(f"{prefix}_API_KEY")
    if provider == "gemini":
        env_key = env_key or _env("GOOGLE_API_KEY")

    defaults = _PROVIDER_DEFAULTS[provider]
    if max_tokens is None:
        max_tokens = max_completion_tokens

    return LLMConfig(
        provider=provider,
        model=model or _env(f"{prefix}_MODEL") or DEFAULT_MODELS[provider],
        api_key=api_key or env_key,
        base_url=base_url or _env(f"{prefix}_BASE_URL"),
        organization=(organization or _env("OPENAI_ORGANIZATION")) if provider == "openai" else None,
        reasoning_effort=reasoning_effort,
        stream=False if stream is None else stream,
        temperature=defaults["temperature"] if temperature is None else temperature,
        top_p=defaults["top_p"] if top_p is None else top_p,
        max_tokens=defaults["max_tokens"] if max_tokens is None else max_tokens,
        extra_body=_parse_extra_body(
            extra_body if extra_body is not None else _env(f"{prefix}_EXTRA_BODY"),
            source=f"{prefix}_EXTRA_BODY" if extra_body is None else "--extra-body",
        ),
        refusal_fallbacks=(_env("ANTHROPIC_FALLBACKS") or "default").lower() not in {"off", "0", "false", "no", "none"},
    )


# Backwards-compatible name used by the benchmark runners.
load_openai_config = load_llm_config


def add_llm_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the LLM endpoint and sampling options shared by the CLI and the benchmark runners."""
    group = parser.add_argument_group("LLM endpoint (unset options come from the environment or a .env file)")
    group.add_argument("--provider", choices=PROVIDERS, default=None, help="LLM provider (default: env LLM_PROVIDER, else openai)")
    group.add_argument("--model", default=None, help="Model name (default: env <PROVIDER>_MODEL, e.g. ANTHROPIC_MODEL)")
    group.add_argument("--base-url", default=None, help="API base URL (default: env <PROVIDER>_BASE_URL)")
    group.add_argument("--api-key", default=None, help="API key (default: env <PROVIDER>_API_KEY)")
    group.add_argument("--extra-body", default=None, help="Extra JSON fields for every request (default: env <PROVIDER>_EXTRA_BODY)")
    group.add_argument(
        "--env-file",
        default=None,
        help="Path to a .env file (default: .env in the current directory or a parent, then the repository root)",
    )
    group.add_argument(
        "--reasoning-effort",
        default=None,
        help="OpenAI reasoning_effort, Anthropic effort (default: high on claude-opus-5-5), or Gemini thinking_level",
    )
    group.add_argument("--temperature", type=float, default=None, help="Sampling temperature (default: 0 for OpenAI, else the API default)")
    group.add_argument("--top-p", type=float, default=None, help="Nucleus sampling top_p (default: 1 for OpenAI, else the API default)")
    group.add_argument(
        "--max-completion-tokens",
        "--max-tokens",
        dest="max_completion_tokens",
        type=int,
        default=None,
        help="Max output tokens per call (default: 4096 for OpenAI, 16000 for Anthropic, the model's limit for Gemini)",
    )
    group.add_argument("--stream", action="store_true", help="Enable streaming model output")


def llm_config_from_args(args: argparse.Namespace) -> LLMConfig:
    return load_llm_config(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        extra_body=args.extra_body,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_completion_tokens,
        stream=args.stream,
        env_file=args.env_file,
    )

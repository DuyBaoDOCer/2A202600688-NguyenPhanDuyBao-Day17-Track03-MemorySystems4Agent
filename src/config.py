from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    confidence_threshold: float          # bonus: min confidence to write to User.md
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load env vars and return a populated LabConfig.

    Priority: .env file in repo root, then environment variables.
    Default provider is mistral; set LLM_PROVIDER to override.
    """
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Load .env from repo root if present
    env_file = root / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file, override=False)
        except ImportError:
            pass

    provider = normalize_provider(os.getenv("LLM_PROVIDER", "custom"))
    model_name = os.getenv("LLM_MODEL", "mistral-small-latest")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    api_key: str | None = None
    base_url: str | None = None

    _key_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "custom": "CUSTOM_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    if provider in _key_map:
        api_key = os.getenv(_key_map[provider])
    if provider == "custom":
        base_url = os.getenv("CUSTOM_BASE_URL", "https://ai-gateway.antco.ai/v1")
    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Judge model (used for response-quality scoring if enabled)
    judge_provider = normalize_provider(os.getenv("JUDGE_LLM_PROVIDER", provider))
    judge_model_name = os.getenv("JUDGE_LLM_MODEL", model_name)
    judge_api_key = os.getenv("JUDGE_API_KEY", api_key)

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )
    judge_cfg = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=0.0,
        api_key=judge_api_key,
    )

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD", "800")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "4")),
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.6")),
        model=model_cfg,
        judge_model=judge_cfg,
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Thin wrapper around the raw openai SDK for any OpenAI-compatible endpoint.
# Used when langchain-openai is not installed.
# ---------------------------------------------------------------------------

class _OpenAICompatModel:
    """Minimal LangChain-compatible wrapper using the raw `openai` SDK."""

    def __init__(self, api_key: str, base_url: str, model: str, temperature: float) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature

    def invoke(self, messages: list[Any]) -> Any:
        # Convert LangChain message objects → OpenAI dict format
        oai_msgs = []
        for m in messages:
            cls = type(m).__name__
            if cls == "SystemMessage":
                role = "system"
            elif cls in ("HumanMessage", "UserMessage"):
                role = "user"
            elif cls in ("AIMessage", "AssistantMessage"):
                role = "assistant"
            else:
                role = "user"
            oai_msgs.append({"role": role, "content": m.content})

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=oai_msgs,
            temperature=self._temperature,
        )

        content = resp.choices[0].message.content or ""
        usage = resp.usage

        class _FakeAIMessage:
            def __init__(self, text: str, u: Any) -> None:
                self.content = text
                self.usage_metadata = {
                    "input_tokens": u.prompt_tokens if u else 0,
                    "output_tokens": u.completion_tokens if u else 0,
                }

        return _FakeAIMessage(content, usage)


@dataclass
class ProviderConfig:
    """Provider configuration shared by both agents.

    Supported providers: mistral, openai, custom, gemini, anthropic, ollama, openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


_ALIASES: dict[str, str] = {
    "anthorpic": "anthropic",
    "open_ai": "openai",
    "openai-compatible": "custom",
    "mistralai": "mistral",
    "google": "gemini",
    "google-genai": "gemini",
}


def normalize_provider(value: str) -> str:
    v = value.strip().lower()
    return _ALIASES.get(v, v)


def build_chat_model(config: ProviderConfig):
    """Instantiate the chat model for the selected provider."""
    provider = normalize_provider(config.provider)

    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        kwargs = dict(model=config.model_name, temperature=config.temperature)
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatMistralAI(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = dict(model=config.model_name, temperature=config.temperature)
        if config.api_key:
            kwargs["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
            kwargs: dict[str, Any] = dict(model=config.model_name, temperature=config.temperature)
            if config.api_key:
                kwargs["api_key"] = config.api_key
            return ChatOpenAI(**kwargs)
        except ImportError:
            return _OpenAICompatModel(
                api_key=config.api_key or "none",
                base_url="https://api.openai.com/v1",
                model=config.model_name,
                temperature=config.temperature,
            )

    if provider == "custom":
        base_url = config.base_url or "https://ai-gateway.antco.ai/v1"
        api_key = config.api_key or "none"
        # Prefer langchain-openai if installed; fall back to raw openai SDK
        try:
            from langchain_openai import ChatOpenAI
            kwargs: dict[str, Any] = dict(model=config.model_name, temperature=config.temperature)
            kwargs["api_key"] = api_key
            kwargs["base_url"] = base_url
            return ChatOpenAI(**kwargs)
        except ImportError:
            return _OpenAICompatModel(
                api_key=api_key,
                base_url=base_url,
                model=config.model_name,
                temperature=config.temperature,
            )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        kwargs = dict(model=config.model_name, temperature=config.temperature)
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatAnthropic(**kwargs)

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url or "http://localhost:11434",
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    raise ValueError(f"Unknown provider: '{provider}'. Supported: mistral, gemini, openai, custom, anthropic, ollama, openrouter")

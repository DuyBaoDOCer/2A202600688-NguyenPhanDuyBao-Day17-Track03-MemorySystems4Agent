from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


_SYSTEM_PROMPT = (
    "Bạn là AI assistant thông minh. Trả lời ngắn gọn và hữu ích bằng tiếng Việt."
)
_SYSTEM_TOKENS = estimate_tokens(_SYSTEM_PROMPT)


class BaselineAgent:
    """Agent A — within-session memory only.

    No persistent storage: facts are forgotten when a new thread_id is used.
    This is the baseline against which AdvancedAgent is measured.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None
        if not force_offline:
            self._maybe_build_langchain_agent()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None and not self.force_offline:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0  # baseline never compacts

    # ------------------------------------------------------------------
    # Offline path (deterministic, no API calls)
    # ------------------------------------------------------------------

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._get_or_create_session(thread_id)

        # Prompt tokens = system + all prior messages + this message
        prior_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        prompt_tokens = _SYSTEM_TOKENS + prior_tokens + estimate_tokens(message)

        # Try to recall facts from the CURRENT thread only
        reply_text = self._recall_from_thread(session, message)

        # Token accounting
        agent_tokens = estimate_tokens(reply_text)
        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": reply_text})
        session.token_usage += agent_tokens
        session.prompt_tokens_processed += prompt_tokens

        return {"reply": reply_text, "tokens": agent_tokens, "prompt_tokens": prompt_tokens}

    def _recall_from_thread(self, session: SessionState, question: str) -> str:
        """Try to answer from facts mentioned earlier in the same thread."""
        if not session.messages:
            return "Xin chào! Bạn cần hỏi gì?"

        # Collect all user text from this thread
        thread_text = " ".join(
            m["content"] for m in session.messages if m["role"] == "user"
        )
        ql = question.lower()

        # Look for keywords that indicate recall questions
        hits: list[str] = []

        if any(k in ql for k in ["tên", "gọi", "là ai"]):
            import re
            m = re.search(r'(?:mình|tôi|tao)\s+tên\s+(?:là\s+)?(\S+)', thread_text)
            if m:
                hits.append(f"Tên bạn là {m.group(1).rstrip('.,!?')}.")

        if any(k in ql for k in ["ở đâu", "nơi ở", "thành phố"]):
            import re
            m = re.search(
                r'(?:mình|tôi)\s+(?:đang\s+)?ở\s+([A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ]\w+(?:\s+[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ]\w+)?)',
                thread_text,
            )
            if m:
                hits.append(f"Bạn đang ở {m.group(1).rstrip('.,!?')}.")

        if any(k in ql for k in ["nghề", "làm gì", "công việc"]):
            import re
            m = re.search(
                r'(?:mình|tôi)\s+(?:đang\s+)?(?:là|làm)\s+(\w+(?:\s+\w+)?\s+(?:engineer|developer|manager))',
                thread_text,
                re.I,
            )
            if m:
                hits.append(f"Nghề nghiệp của bạn là {m.group(1).rstrip('.,!?')}.")

        if any(k in ql for k in ["đồ uống", "uống gì"]):
            import re
            m = re.search(r'(?:đồ\s+uống\s+yêu\s+thích\s+(?:là|của\s+mình\s+là))\s+(.+?)(?:\.|$)', thread_text, re.I)
            if m:
                hits.append(f"Đồ uống yêu thích của bạn là {m.group(1).rstrip('.,!?')}.")

        if hits:
            return " ".join(hits)

        return "Xin lỗi, mình chỉ nhớ được thông tin trong cuộc trò chuyện hiện tại."

    # ------------------------------------------------------------------
    # Live path (Mistral / other real LLM)
    # ------------------------------------------------------------------

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        from langchain_core.messages import AIMessage as LCAIMessage
        from langchain_core.messages import HumanMessage, SystemMessage

        session = self._get_or_create_session(thread_id)

        msgs: list[Any] = [SystemMessage(content=_SYSTEM_PROMPT)]
        for m in session.messages:
            if m["role"] == "user":
                msgs.append(HumanMessage(content=m["content"]))
            else:
                msgs.append(LCAIMessage(content=m["content"]))
        msgs.append(HumanMessage(content=message))

        prompt_tokens_est = sum(estimate_tokens(m.content) for m in msgs)

        response = self.langchain_agent.invoke(msgs)
        reply_text = response.content

        input_tok, output_tok = _parse_usage(response, prompt_tokens_est, reply_text)

        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": reply_text})
        session.token_usage += output_tok
        session.prompt_tokens_processed += input_tok

        return {"reply": reply_text, "tokens": output_tok, "prompt_tokens": input_tok}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_session(self, thread_id: str) -> SessionState:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def _maybe_build_langchain_agent(self) -> None:
        try:
            self.langchain_agent = build_chat_model(self.config.model)
        except Exception:
            self.langchain_agent = None


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _parse_usage(response: Any, fallback_input: int, fallback_output_text: str) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage:
        return usage.get("input_tokens", fallback_input), usage.get("output_tokens", estimate_tokens(fallback_output_text))
    meta = getattr(response, "response_metadata", {}) or {}
    for key in ("usage", "token_usage"):
        if key in meta:
            u = meta[key]
            return u.get("prompt_tokens", fallback_input), u.get("completion_tokens", estimate_tokens(fallback_output_text))
    return fallback_input, estimate_tokens(fallback_output_text)

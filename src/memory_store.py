from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Heuristic: ~4 chars per token, consistent with common LLM tokenizers."""
    text = text.strip()
    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# UserProfileStore  —  persistent User.md per user_id
# ---------------------------------------------------------------------------

_DEFAULT_PROFILE = "# User Profile\n\n"


@dataclass
class UserProfileStore:
    """Maps each user_id to one Markdown file under root_dir."""

    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, user_id: str) -> Path:
        safe = re.sub(r"[^\w\-]", "_", user_id)
        return self.root_dir / f"{safe}.md"

    def read_text(self, user_id: str) -> str:
        p = self.path_for(user_id)
        if not p.exists():
            return _DEFAULT_PROFILE
        return p.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        p = self.path_for(user_id)
        p.write_text(content, encoding="utf-8")
        return p

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        p = self.path_for(user_id)
        return p.stat().st_size if p.exists() else 0

    # --- Structured helpers -------------------------------------------------

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse User.md into a key→value dict (one value per ## section)."""
        result: dict[str, str] = {}
        current_key: str | None = None
        for line in self.read_text(user_id).split("\n"):
            line = line.strip()
            if line.startswith("## "):
                current_key = line[3:].strip()
            elif line and current_key and not line.startswith("#"):
                result[current_key] = line
        return result

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        """Insert or replace a ## section in User.md."""
        text = self.read_text(user_id)
        # Try to replace existing section value
        pattern = re.compile(
            rf"(^## {re.escape(key)}\n)(.*?)(?=\n## |\Z)",
            re.MULTILINE | re.DOTALL,
        )
        if pattern.search(text):
            text = pattern.sub(rf"\g<1>{value}\n", text)
        else:
            text = text.rstrip("\n") + f"\n\n## {key}\n{value}\n"
        self.write_text(user_id, text)


# ---------------------------------------------------------------------------
# extract_profile_updates  (BONUS: confidence threshold)
# ---------------------------------------------------------------------------

# Confidence scores per extraction pattern:
#   0.90–0.95  explicit, first-person declarations  ("mình tên là X")
#   0.80–0.89  corrections / high-signal rewrites   ("giờ mình đang ở X")
#   0.70–0.79  general first-person claims          ("mình ở X")
#   0.40–0.59  implicit / incidental mentions       (Python/AI in passing)
#
# Default threshold 0.6 → accepts everything except implicit interests (0.5).
# Raising threshold to 0.8 → only very explicit, high-signal facts land in User.md,
# reducing false positives at the cost of occasionally missing indirect updates.
# Risk: a too-high threshold silently drops real facts if phrased informally.

def extract_profile_updates(message: str, min_confidence: float = 0.0) -> dict[str, str]:
    """Extract stable profile facts from a user message.

    Args:
        message: Raw user message text.
        min_confidence: Minimum confidence score [0, 1] for a fact to be
            included.  0.0 keeps everything; 0.6 (default in LabConfig)
            filters implicit/uncertain mentions; 0.85+ keeps only explicit
            high-signal declarations and documented corrections.

    Returns:
        {fact_key: fact_value} for facts whose confidence >= min_confidence.
    """
    # Internal: scored facts before threshold filtering
    scored: dict[str, tuple[str, float]] = {}  # key → (value, confidence)

    # Skip pure-question turns (no factual declaration expected)
    q = message.count("?")
    sentences_approx = max(1, len(re.findall(r"[.!]", message)) + 1)
    if q > 0 and q >= sentences_approx and len(message) < 160:
        return {}

    raw_sentences = re.split(r"(?<=[.!])\s+", message.strip())

    # --- Name  (confidence 0.95 — explicit first-person declaration) ---
    for sent in raw_sentences:
        m = re.search(
            r"(?:mình|tôi|tao)\s+tên\s+(?:là\s+)?"
            r"([A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ]\w+(?:\s+[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ]\w+)?)",
            sent,
        )
        if not m:
            m = re.search(
                r"tên\s+(?:mình|tôi)\s+là\s+"
                r"([A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ]\w+(?:\s+[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ]\w+)?)",
                sent,
            )
        if m:
            name = m.group(1).rstrip(".,!?:")
            if name.lower() not in {"gì", "ai", "không", "bạn", "là", "ở"}:
                scored["name"] = (name, 0.95)
                break

    # --- Location ---
    loc_value: str | None = None
    loc_conf: float = 0.70

    # Correction patterns → confidence 0.88 (explicit update)
    correction_pats = [
        r"(?:thực\s+ra|đính\s+chính)[^\n,;]*?\bở\s+"
        r"([A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ]\w+(?:\s+[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ]\w+)?)",
        r"(?:giờ|hiện(?:\s+tại)?|bây\s+giờ)\s+[^\n,;]{0,30}?\bở\s+"
        r"([A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ]\w+(?:\s+[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ]\w+)?)",
        r"đang\s+làm\s+việc\s+ở\s+"
        r"([A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ]\w+(?:\s+[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ]\w+)?)",
    ]
    for pat in correction_pats:
        m = re.search(pat, message)
        if m:
            loc_value = m.group(1).split()[0].rstrip(".,!?")
            loc_conf = 0.88
            break

    if loc_value is None:
        # General claim → confidence 0.75
        positive_parts: list[str] = []
        for sent in raw_sentences:
            sl = sent.lower()
            if any(neg in sl for neg in ["không còn ở", "chỉ là nơi", "không phải nơi", "chỉ đùa"]):
                continue
            part = re.split(r"\bchứ\b", sent)[0]
            positive_parts.append(part)
        pos_text = " ".join(positive_parts)
        for pat in [
            r"(?:mình|tôi)\s+(?:đang\s+)?ở\s+"
            r"([A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ]\w+(?:\s+[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ]\w+)?)",
            r"(?:mình|tôi)\s+(?:hiện\s+)?sống\s+(?:ở|tại)\s+"
            r"([A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ]\w+(?:\s+[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ]\w+)?)",
        ]:
            m = re.search(pat, pos_text)
            if m:
                loc_value = m.group(1).split()[0].rstrip(".,!?")
                loc_conf = 0.75
                break

    if loc_value:
        _SKIP_LOCS = {"Không", "Còn", "Rồi", "Nữa", "Đây", "Đó", "Đâu", "Này"}
        if loc_value not in _SKIP_LOCS and len(loc_value) >= 2:
            scored["location"] = (loc_value, loc_conf)

    # --- Profession ---
    prof_sentences: list[str] = []
    for sent in raw_sentences:
        sl = sent.lower()
        if "câu đùa" in sl or "chỉ là câu đùa" in sl or "nhưng đó chỉ là" in sl:
            continue
        if "đùa" in sl and re.search(
            r"(?:chuyển\s+sang|là)\s+\w+\s+(?:manager|engineer|developer)", sent, re.I
        ):
            continue
        prof_sentences.append(sent)
    prof_text = " ".join(prof_sentences)

    prof_conf_map = [
        # (pattern, confidence)
        (r"(?:chuyển\s+sang)\s+(\w+(?:\s+\w+)?\s+(?:engineer|developer|manager|designer|analyst|scientist))", 0.88),
        (r"(?:nghề\s+nghiệp\s+(?:hiện\s+tại\s+)?(?:vẫn\s+)?(?:là|làm))\s+"
         r"(\w+(?:\s+\w+)?\s+(?:engineer|developer|manager|designer|analyst|scientist))", 0.88),
        (r"(?:mình|tôi)\s+(?:đang\s+)?(?:là|làm)\s+"
         r"(\w+(?:\s+\w+)?\s+(?:engineer|developer|manager|designer|analyst|scientist))\b", 0.78),
    ]
    for pat, conf in prof_conf_map:
        m = re.search(pat, prof_text, re.I)
        if m:
            scored["profession"] = (m.group(1).strip().rstrip(".,!?"), conf)
            break

    # --- Drink  (confidence 0.92 — very explicit pattern required) ---
    for pat in [
        r"(?:đồ\s+uống\s+yêu\s+thích\s+(?:(?:của\s+mình\s+)?là))\s+(.+?)(?:\.|$)",
        r"(?:uống|thích\s+uống)\s+(cà\s+phê[\w\s]{0,20})",
    ]:
        m = re.search(pat, message, re.I)
        if m:
            scored["drink"] = (m.group(1).strip().rstrip(".,!?"), 0.92)
            break

    # --- Food  (confidence 0.92) ---
    m = re.search(
        r"(?:món\s+ăn\s+yêu\s+thích\s+(?:(?:của\s+mình\s+)?là))\s+(.+?)(?:\.|$)",
        message,
        re.I,
    )
    if m:
        scored["food"] = (m.group(1).strip().rstrip(".,!?"), 0.92)

    # --- Pet  (confidence 0.92) ---
    m = re.search(r"nuôi\s+(?:một\s+)?(?:bé\s+)?(\w+)\s+tên\s+(\w+)", message, re.I)
    if m:
        scored["pet"] = (f"{m.group(1)} tên {m.group(2)}", 0.92)

    # --- Response style  (confidence 0.82) ---
    m = re.search(
        r"(?:muốn|thích)\s+(?:bạn\s+)?(?:câu\s+)?trả\s+lời\s+(.+?)(?:\.|$)",
        message,
        re.I,
    )
    if m:
        style = m.group(1).strip().rstrip(".,!?")
        if len(style) < 140:
            scored["response_style"] = (style, 0.82)
    else:
        m = re.search(
            r"(?:câu\s+trả\s+lời|trả\s+lời)\s+(?:theo\s+dạng\s+)?(\d+\s+bullet[^.]{0,80})",
            message,
            re.I,
        )
        if m:
            scored["response_style"] = (m.group(1).strip().rstrip(".,!?"), 0.82)

    # --- Technical interests  (confidence 0.50 — implicit mention) ---
    interests: list[str] = []
    ml = message.lower()
    if "python" in ml:
        interests.append("Python")
    if re.search(r"\bai\b|\bai\s+ứng\s+dụng\b", ml):
        interests.append("AI ứng dụng")
    if "mlops" in ml:
        interests.append("MLOps")
    if "rag" in ml:
        interests.append("RAG")
    if interests and "interests" not in scored:
        scored["interests"] = (", ".join(interests), 0.50)

    # Apply threshold filter
    return {k: v for k, (v, conf) in scored.items() if conf >= min_confidence}


# ---------------------------------------------------------------------------
# summarize_messages — heuristic summary for compact memory
# ---------------------------------------------------------------------------

def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    if not messages:
        return ""
    kept = messages[-max_items:] if len(messages) > max_items else messages
    lines = [f"[Tóm tắt {len(messages)} tin nhắn trước:]"]
    for msg in kept:
        role = "Người dùng" if msg.get("role") == "user" else "Trợ lý"
        content = msg.get("content", "")
        if len(content) > 120:
            content = content[:117] + "..."
        lines.append(f"- {role}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CompactMemoryManager
# ---------------------------------------------------------------------------

@dataclass
class CompactMemoryManager:
    """Manages per-thread conversation history with automatic compaction.

    When the token count of a thread exceeds threshold_tokens, the oldest
    messages are summarised and replaced with a compact summary block.
    keep_messages recent messages are always preserved in full.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def _init_thread(self, thread_id: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }

    def append(self, thread_id: str, role: str, content: str) -> None:
        self._init_thread(thread_id)
        self.state[thread_id]["messages"].append({"role": role, "content": content})  # type: ignore[index]
        self._maybe_compact(thread_id)

    def context(self, thread_id: str) -> dict[str, object]:
        self._init_thread(thread_id)
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        self._init_thread(thread_id)
        return int(self.state[thread_id]["compactions"])  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    def _total_tokens(self, thread_id: str) -> int:
        s = self.state[thread_id]
        msg_tokens = sum(estimate_tokens(m["content"]) for m in s["messages"])  # type: ignore[union-attr]
        summary_tokens = estimate_tokens(str(s["summary"]))
        return msg_tokens + summary_tokens

    def _maybe_compact(self, thread_id: str) -> None:
        if self._total_tokens(thread_id) <= self.threshold_tokens:
            return
        msgs: list[dict[str, str]] = self.state[thread_id]["messages"]  # type: ignore[assignment]
        if len(msgs) <= self.keep_messages:
            return
        old = msgs[: -self.keep_messages]
        recent = msgs[-self.keep_messages :]
        prev_summary = str(self.state[thread_id]["summary"])
        new_parts = []
        if prev_summary:
            new_parts.append(prev_summary)
        new_parts.append(summarize_messages(old))
        self.state[thread_id]["summary"] = "\n".join(new_parts)
        self.state[thread_id]["messages"] = recent  # type: ignore[assignment]
        self.state[thread_id]["compactions"] = int(self.state[thread_id]["compactions"]) + 1  # type: ignore[assignment]

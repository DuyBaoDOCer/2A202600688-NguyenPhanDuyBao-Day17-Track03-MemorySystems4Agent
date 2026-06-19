from __future__ import annotations

from pathlib import Path

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens
from model_provider import ProviderConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated LabConfig pointing at tmp_path for state storage."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    dummy_model = ProviderConfig(
        provider="mistral",
        model_name="mistral-small-latest",
        temperature=0.0,
    )
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=60,   # low threshold so compaction triggers quickly
        compact_keep_messages=2,
        confidence_threshold=0.0,      # no filtering in tests — capture everything
        model=dummy_model,
        judge_model=dummy_model,
    )


# ---------------------------------------------------------------------------
# Test 1: UserProfileStore read / write / edit
# ---------------------------------------------------------------------------

def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")

    user_id = "test_user"

    # Default read returns starter markdown
    text = store.read_text(user_id)
    assert "User Profile" in text

    # Write then read back
    content = "# User Profile\n\n## name\nAlice\n\n## location\nHà Nội\n"
    store.write_text(user_id, content)
    assert store.read_text(user_id) == content

    # Edit replaces one occurrence
    changed = store.edit_text(user_id, "Alice", "Bob")
    assert changed is True
    assert "Bob" in store.read_text(user_id)
    assert "Alice" not in store.read_text(user_id)

    # Edit returns False when search text not found
    changed = store.edit_text(user_id, "nonexistent", "x")
    assert changed is False

    # file_size reflects written bytes
    assert store.file_size(user_id) > 0

    # upsert_fact creates new section
    store.upsert_fact(user_id, "profession", "data scientist")
    assert "data scientist" in store.read_text(user_id)

    # upsert_fact overwrites existing section
    store.upsert_fact(user_id, "profession", "ML engineer")
    text = store.read_text(user_id)
    assert "ML engineer" in text
    assert "data scientist" not in text

    # facts() parses all sections back
    f = store.facts(user_id)
    assert f["name"] == "Bob"
    assert f["location"] == "Hà Nội"
    assert f["profession"] == "ML engineer"


# ---------------------------------------------------------------------------
# Test 2: CompactMemoryManager triggers compaction
# ---------------------------------------------------------------------------

def test_compact_trigger(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    mem = CompactMemoryManager(
        threshold_tokens=cfg.compact_threshold_tokens,
        keep_messages=cfg.compact_keep_messages,
    )

    thread = "t1"
    # Each message ≈ 25 tokens (100 chars).  Threshold = 60 tokens.
    long_msg = "x" * 100  # 100 chars => ~25 tokens

    assert mem.compaction_count(thread) == 0

    for i in range(6):
        mem.append(thread, "user", f"Message {i}: {long_msg}")
        mem.append(thread, "assistant", f"Reply {i}: ok")

    assert mem.compaction_count(thread) >= 1, (
        "Compaction should have triggered after many long messages"
    )

    ctx = mem.context(thread)
    assert ctx["summary"], "Summary should be non-empty after compaction"
    assert len(ctx["messages"]) <= cfg.compact_keep_messages, (  # type: ignore[arg-type]
        "Recent messages should be trimmed to keep_messages"
    )


# ---------------------------------------------------------------------------
# Test 3: cross-session recall — Advanced remembers, Baseline forgets
# ---------------------------------------------------------------------------

def test_cross_session_recall(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)

    # --- Advanced agent ---
    adv = AdvancedAgent(config=cfg, force_offline=True)

    # Session 1: provide facts
    adv.reply("alice", "session-1", "Chào bạn, mình tên là Alice.")
    adv.reply("alice", "session-1", "Mình ở Hà Nội và đang làm data scientist.")
    adv.reply("alice", "session-1", "Đồ uống yêu thích là trà sữa.")

    # Session 2 (NEW thread): recall
    r = adv.reply("alice", "session-2", "Mình tên gì vậy?")
    assert "Alice" in r["reply"], (
        f"Advanced should recall name across sessions; got: {r['reply']!r}"
    )

    r2 = adv.reply("alice", "session-2", "Đồ uống yêu thích của mình là gì?")
    assert "trà sữa" in r2["reply"], (
        f"Advanced should recall drink across sessions; got: {r2['reply']!r}"
    )

    # --- Baseline agent ---
    base = BaselineAgent(config=cfg, force_offline=True)

    base.reply("alice", "session-1", "Chào bạn, mình tên là Alice.")
    base.reply("alice", "session-1", "Mình ở Hà Nội và đang làm data scientist.")

    # Different thread — baseline must NOT recall
    r3 = base.reply("alice", "session-2", "Mình tên gì vậy?")
    # Baseline gives a generic or incorrect answer
    assert "Alice" not in r3["reply"], (
        f"Baseline should NOT recall name in a new thread; got: {r3['reply']!r}"
    )


# ---------------------------------------------------------------------------
# Test 4: compact memory reduces cumulative prompt load on a long thread
# ---------------------------------------------------------------------------

def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)

    baseline = BaselineAgent(config=cfg, force_offline=True)
    advanced = AdvancedAgent(config=cfg, force_offline=True)

    user_id = "bob"
    thread = "long-thread"
    long_msg = "Đây là một tin nhắn khá dài để kiểm tra compact memory. " * 4

    # Feed 12 long messages to both agents
    for i in range(12):
        baseline.reply(user_id, thread, f"Turn {i}: {long_msg}")
        advanced.reply(user_id, thread, f"Turn {i}: {long_msg}")

    base_prompt = baseline.prompt_token_usage(thread)
    adv_prompt = advanced.prompt_token_usage(thread)

    # Advanced should have triggered compaction
    assert advanced.compaction_count(thread) >= 1, (
        "Advanced should have compacted the long thread at least once"
    )

    # Advanced cumulative prompt load should be less than baseline
    # (compact prevents unbounded context growth)
    assert adv_prompt < base_prompt, (
        f"Advanced prompt tokens ({adv_prompt}) should be < Baseline ({base_prompt}) "
        "after compaction kicks in on a long thread"
    )

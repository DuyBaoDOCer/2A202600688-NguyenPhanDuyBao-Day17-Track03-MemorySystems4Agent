from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_conversations(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def recall_points(answer: str, expected: list[str]) -> float:
    """0 / 0.5 / 1.0 based on how many expected keywords appear in the answer."""
    if not expected:
        return 1.0
    al = answer.lower()
    hits = sum(1 for kw in expected if kw.lower() in al)
    if hits == len(expected):
        return 1.0
    if hits > 0:
        return hits / len(expected)
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score: keyword coverage + response length."""
    if not answer.strip():
        return 0.0
    al = answer.lower()
    hits = sum(1 for kw in expected if kw.lower() in al)
    keyword_ratio = hits / max(1, len(expected))
    length_score = min(1.0, len(answer) / 80)
    return round(0.7 * keyword_ratio + 0.3 * length_score, 2)


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_agent_benchmark(
    agent_name: str,
    agent: Any,
    conversations: list[dict[str, Any]],
    config: Any,
) -> BenchmarkRow:
    """Evaluate one agent over a list of conversations.

    For each conversation:
    1. Process all turns in the main thread.
    2. Ask recall questions in a fresh (different) thread.
    3. Collect token counts, recall scores, and quality scores.
    """
    total_agent_tokens = 0
    total_prompt_tokens = 0
    total_recall = 0.0
    total_quality = 0.0
    total_rq_count = 0
    total_compactions = 0
    memory_growth_bytes = 0
    seen_users: set[str] = set()

    for conv in conversations:
        user_id: str = conv["user_id"]
        main_thread = conv["id"] + "_main"
        recall_thread = conv["id"] + "_recall"

        # --- Feed all turns ---
        for turn in conv.get("turns", []):
            result = agent.reply(user_id, main_thread, turn)
            total_agent_tokens += result.get("tokens", 0)
            total_prompt_tokens += result.get("prompt_tokens", 0)

        # --- Ask recall questions in a new thread ---
        for rq in conv.get("recall_questions", []):
            question: str = rq["question"]
            expected: list[str] = rq.get("expected_contains", [])

            result = agent.reply(user_id, recall_thread, question)
            answer = result.get("reply", "")
            total_agent_tokens += result.get("tokens", 0)
            total_prompt_tokens += result.get("prompt_tokens", 0)

            total_recall += recall_points(answer, expected)
            total_quality += heuristic_quality(answer, expected)
            total_rq_count += 1

        total_compactions += agent.compaction_count(main_thread)
        total_compactions += agent.compaction_count(recall_thread)
        seen_users.add(user_id)

    # Measure persistent memory growth (advanced agent only)
    if hasattr(agent, "memory_file_size"):
        for uid in seen_users:
            memory_growth_bytes += agent.memory_file_size(uid)

    n = max(1, total_rq_count)
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(total_recall / n, 3),
        response_quality=round(total_quality / n, 3),
        memory_growth_bytes=memory_growth_bytes,
        compactions=total_compactions,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_rows(rows: list[BenchmarkRow]) -> str:
    from tabulate import tabulate

    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    data = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score:.3f}",
            f"{r.response_quality:.3f}",
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]
    return tabulate(data, headers=headers, tablefmt="github")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)

    std_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    std_convs = load_conversations(std_path)
    stress_convs = load_conversations(stress_path)

    print("=" * 70)
    print("STANDARD BENCHMARK  (data/conversations.json)")
    print("=" * 70)

    baseline_std = BaselineAgent(config=config, force_offline=True)
    advanced_std = AdvancedAgent(config=config, force_offline=True)

    rows_std = [
        run_agent_benchmark("Baseline", baseline_std, std_convs, config),
        run_agent_benchmark("Advanced", advanced_std, std_convs, config),
    ]
    print(format_rows(rows_std))

    print()
    _print_analysis_standard(rows_std)

    print()
    print("=" * 70)
    print("LONG-CONTEXT STRESS BENCHMARK  (data/advanced_long_context.json)")
    print("=" * 70)

    baseline_stress = BaselineAgent(config=config, force_offline=True)
    advanced_stress = AdvancedAgent(config=config, force_offline=True)

    rows_stress = [
        run_agent_benchmark("Baseline", baseline_stress, stress_convs, config),
        run_agent_benchmark("Advanced", advanced_stress, stress_convs, config),
    ]
    print(format_rows(rows_stress))

    print()
    _print_analysis_stress(rows_stress)


def _print_analysis_standard(rows: list[BenchmarkRow]) -> None:
    """
    Phân tích standard benchmark (nhiều cuộc trò chuyện ngắn, nhiều thread).

    Câu hỏi cốt lõi cần giải thích:
      1. Tại sao Advanced recall cao hơn nhiều?
      2. Compact memory có giúp ở hội thoại ngắn không?
      3. Memory growth và rủi ro dài hạn là gì?
    """
    print("PHÂN TÍCH STANDARD BENCHMARK")
    print("-" * 50)
    if len(rows) < 2:
        return
    b, a = rows[0], rows[1]

    # --- Cross-session recall ---
    print(f"• Cross-session recall — Baseline: {b.recall_score:.3f}  Advanced: {a.recall_score:.3f}")
    recall_gain = a.recall_score - b.recall_score
    if recall_gain > 0.5:
        print(
            f"  → Advanced recall cao hơn {recall_gain:.2f} điểm nhờ User.md lưu facts\n"
            "     bền vững qua nhiều thread khác nhau. Baseline chỉ có within-session\n"
            "     memory nên quên hoàn toàn khi thread mới bắt đầu."
        )
    elif recall_gain > 0:
        print("  → Advanced recall cao hơn nhờ User.md — nhưng mức gain nhỏ.")

    # --- Prompt tokens ---
    print(f"• Prompt tokens — Baseline: {b.prompt_tokens_processed}  Advanced: {a.prompt_tokens_processed}")
    if a.prompt_tokens_processed < b.prompt_tokens_processed:
        savings_pct = (b.prompt_tokens_processed - a.prompt_tokens_processed) / b.prompt_tokens_processed * 100
        print(
            f"  → Advanced tiết kiệm {savings_pct:.0f}% prompt tokens dù phải mang thêm\n"
            "     User.md vào system prompt. Lý do: compact memory tóm tắt lịch sử cũ\n"
            "     nên context mỗi lượt không tăng tuyến tính như Baseline.\n"
            f"     Compactions: {a.compactions} lần — mỗi lần nén giữ lại chỉ {b.compactions} tin nhắn gần nhất."
        )
    elif a.prompt_tokens_processed > b.prompt_tokens_processed:
        overhead = a.prompt_tokens_processed - b.prompt_tokens_processed
        print(
            f"  → Advanced tốn thêm {overhead} prompt tokens vì mang theo User.md và\n"
            "     compact summary. Với hội thoại rất ngắn (1-2 lượt), overhead này\n"
            "     chưa được bù đắp bởi compaction — compact chỉ có lợi khi lịch sử\n"
            "     đủ dài để vượt threshold."
        )
    else:
        print("  → Prompt tokens tương đương.")

    # --- Memory growth ---
    print(f"• Memory growth — Baseline: {b.memory_growth_bytes} bytes  Advanced: {a.memory_growth_bytes} bytes")
    if a.memory_growth_bytes > 0:
        print(
            "  → User.md phát triển theo thời gian. Rủi ro: facts cũ không tự xóa;\n"
            "     nếu user cung cấp thông tin sai hoặc đùa giỡn, có thể ô nhiễm profile.\n"
            "     Confidence threshold (mặc định 0.6) giúp lọc các extraction không chắc."
        )

    # --- Compactions ---
    print(f"• Compactions — Advanced: {a.compactions}")
    if a.compactions > 10:
        print(
            "  → Số lần compact cao chứng tỏ ngưỡng token bị vượt thường xuyên, nhất\n"
            "     là với LLM live (phản hồi dài hơn offline). Trade-off: compact tiết\n"
            "     kiệm prompt load nhưng có thể mất chi tiết trong các turns đã nén."
        )


def _print_analysis_stress(rows: list[BenchmarkRow]) -> None:
    """
    Phân tích stress benchmark (1 cuộc trò chuyện dài 15+ turns).

    Đây là kịch bản mà compact memory thể hiện rõ nhất lợi thế:
    Baseline phải kéo toàn bộ lịch sử vào context mỗi lượt → O(N²) tokens.
    Advanced nén ngữ cảnh cũ → O(K) tokens trong đó K = số tin nhắn giữ lại.
    """
    print("PHÂN TÍCH STRESS BENCHMARK")
    print("-" * 50)
    if len(rows) < 2:
        return
    b, a = rows[0], rows[1]

    # --- Recall ---
    print(f"• Cross-session recall — Baseline: {b.recall_score:.3f}  Advanced: {a.recall_score:.3f}")
    if a.recall_score >= 1.0 and b.recall_score == 0.0:
        print(
            "  → Advanced: 1.0 | Baseline: 0.0 — kết quả phân kỳ hoàn toàn.\n"
            "     Facts được giới thiệu ở turns đầu không còn trong context của Baseline\n"
            "     khi recall question xuất hiện ở thread khác. Advanced lấy lại từ User.md."
        )

    # --- Prompt tokens: the key O(N²) vs O(K) story ---
    print(f"• Prompt tokens — Baseline: {b.prompt_tokens_processed}  Advanced: {a.prompt_tokens_processed}")
    if b.prompt_tokens_processed > a.prompt_tokens_processed:
        savings_pct = (b.prompt_tokens_processed - a.prompt_tokens_processed) / b.prompt_tokens_processed * 100
        print(
            f"  → Compact memory giúp Advanced giảm {savings_pct:.0f}% prompt tokens.\n"
            "     Baseline tăng theo O(N²): mỗi lượt thứ N phải đọc lại N-1 lượt trước.\n"
            "     Advanced tăng theo O(K): chỉ giữ K tin nhắn gần nhất + 1 đoạn tóm tắt.\n"
            "     Trong hội thoại rất dài (>50 turns), chênh lệch này có thể lên hàng chục\n"
            "     nghìn tokens — tiết kiệm đáng kể cả chi phí lẫn độ trễ."
        )

    # --- Compactions ---
    print(f"• Compactions triggered: {a.compactions}")
    if a.compactions > 0:
        print(
            f"  → {a.compactions} lần compact đã kích hoạt, mỗi lần nén messages cũ thành\n"
            "     summary + giữ lại số tin nhắn gần nhất theo compact_keep_messages.\n"
            "     Rủi ro: nếu compact_keep_messages quá nhỏ, chi tiết quan trọng của\n"
            "     turns trước có thể bị mất trong tóm tắt."
        )

    # --- Bonus summary ---
    print()
    print("BONUS — Confidence Threshold")
    print("-" * 50)
    print(
        "• extract_profile_updates() gán điểm confidence cho mỗi extraction:\n"
        "    0.92–0.95  khai báo tường minh ('mình tên là X', 'đồ uống yêu thích là X')\n"
        "    0.80–0.88  sửa lỗi / cập nhật ('giờ mình đang ở X', 'chuyển sang X engineer')\n"
        "    0.70–0.78  tuyên bố chung ('mình ở X', 'mình là X')\n"
        "    0.50       ngầm định ('Python' xuất hiện trong câu bình thường)\n"
        "\n"
        "• CONFIDENCE_THRESHOLD=0.6 (mặc định) → lọc bỏ implicit interests (0.5),\n"
        "  giữ lại tất cả khai báo rõ ràng. Đặt 0.85 để chỉ giữ explicit + corrections.\n"
        "\n"
        "• Vấn đề giải quyết: không để câu đùa hoặc ngữ cảnh mơ hồ làm ô nhiễm User.md.\n"
        "• Cải thiện metric: recall tăng vì User.md chứa ít false positive hơn.\n"
        "• Rủi ro thêm vào: threshold cao quá → bỏ sót facts thật sự; cần tune theo dataset."
    )


if __name__ == "__main__":
    main()

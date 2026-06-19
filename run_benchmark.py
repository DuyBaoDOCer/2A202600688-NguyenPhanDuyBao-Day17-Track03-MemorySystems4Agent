"""Benchmark runner.

Auto-detects live vs offline mode:
- If CUSTOM_API_KEY (or relevant provider key) is set in .env → live mode (real LLM calls)
- Otherwise → offline mode (deterministic heuristics, no API calls)

Usage:
    conda run -n vinuni_py311 python run_benchmark.py
    conda run -n vinuni_py311 python run_benchmark.py --offline   # force offline
"""
import io
import sys

# Force UTF-8 output so Vietnamese characters display correctly on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, "src")

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import (
    _print_analysis_standard,
    _print_analysis_stress,
    format_rows,
    load_conversations,
    run_agent_benchmark,
)
from config import load_config


def _has_api_key(config) -> bool:
    """Return True if a real API key appears to be configured."""
    return bool(config.model.api_key and config.model.api_key.strip())


def main(force_offline: bool = False) -> None:
    config = load_config(Path(__file__).resolve().parent)

    live = not force_offline and _has_api_key(config)
    mode_label = "LIVE (real LLM)" if live else "OFFLINE (heuristic)"
    print(f"Mode: {mode_label}")
    if live:
        print(f"Provider: {config.model.provider} | Model: {config.model.model_name}")
        if config.model.base_url:
            print(f"Base URL: {config.model.base_url}")
    print()

    std_convs = load_conversations(config.data_dir / "conversations.json")
    stress_convs = load_conversations(config.data_dir / "advanced_long_context.json")

    print("=" * 70)
    print("STANDARD BENCHMARK  (data/conversations.json)")
    print("=" * 70)
    baseline_std = BaselineAgent(config=config, force_offline=not live)
    advanced_std = AdvancedAgent(config=config, force_offline=not live)
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
    baseline_stress = BaselineAgent(config=config, force_offline=not live)
    advanced_stress = AdvancedAgent(config=config, force_offline=not live)
    rows_stress = [
        run_agent_benchmark("Baseline", baseline_stress, stress_convs, config),
        run_agent_benchmark("Advanced", advanced_stress, stress_convs, config),
    ]
    print(format_rows(rows_stress))
    print()
    _print_analysis_stress(rows_stress)


if __name__ == "__main__":
    offline = "--offline" in sys.argv
    main(force_offline=offline)

"""Direct test runner — bypasses pytest plugin loading entirely.

Usage:
    conda run -n vinuni_py311 python run_tests.py
"""
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from test_agents import (
    test_compact_reduces_prompt_load_on_long_thread,
    test_compact_trigger,
    test_cross_session_recall,
    test_user_markdown_read_write_edit,
)

TESTS = [
    test_user_markdown_read_write_edit,
    test_compact_trigger,
    test_cross_session_recall,
    test_compact_reduces_prompt_load_on_long_thread,
]


def run_all() -> int:
    passed = failed = 0
    for fn in TESTS:
        name = fn.__name__
        with tempfile.TemporaryDirectory() as tmp:
            try:
                fn(Path(tmp))
                print(f"  PASS  {name}")
                passed += 1
            except AssertionError as exc:
                print(f"  FAIL  {name}")
                print(f"        {exc}")
                failed += 1
            except Exception:
                print(f"  ERROR {name}")
                traceback.print_exc()
                failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all())

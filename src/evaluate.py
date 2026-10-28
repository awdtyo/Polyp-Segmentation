"""Evaluation entry point - implemented in Milestone 6.

Planned interface (AGENTS.md section 5):

    python -m src.evaluate --config configs/exp001_baseline.yaml --split test

which appends one row to ``results/results.csv``. Nothing is implemented yet;
this module only exists so the repo layout matches AGENTS.md section 7.
"""

from __future__ import annotations

import sys


def main() -> int:
    """Placeholder entry point.

    Returns:
        Exit code 1, because evaluation is not implemented yet.
    """
    print("src/evaluate.py is implemented in Milestone 6 (AGENTS.md section 9).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

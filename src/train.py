"""Training entry point - implemented in Milestone 2.

Planned interface (AGENTS.md section 4):

    python -m src.train --config configs/exp001_baseline.yaml
    python -m src.train --config configs/exp001_baseline.yaml --smoke

``--smoke`` runs a tiny subset, a small image size and a few steps on CPU.
No training logic exists yet; this module only exists so the repo layout matches
AGENTS.md section 7.
"""

from __future__ import annotations

import sys


def main() -> int:
    """Placeholder entry point.

    Returns:
        Exit code 1, because training is not implemented yet.

    Raises:
        SystemExit: never; returns instead.
    """
    print("src/train.py is implemented in Milestone 2 (AGENTS.md section 9).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

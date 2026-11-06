#!/usr/bin/env python
"""Convenience launcher for experiments.

Usage
-----
    python run.py <experiment> [args...]

Examples
--------
    python run.py dmc/walker_sac --seed 1
    python run.py baselines/mt10_sac_v2 --seed 3 --track --wandb-project foo --wandb-entity bar
    python run.py                          # lists available experiments
"""

import subprocess
import sys
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).parent / "experiments"


def _list_experiments() -> list[Path]:
    return sorted(
        p for p in _EXPERIMENTS_DIR.rglob("*.py")
        if not p.name.startswith("_")
    )


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        print("Available experiments:")
        for p in _list_experiments():
            rel = p.relative_to(_EXPERIMENTS_DIR).with_suffix("")
            print(f"  {rel}")
        sys.exit(0)

    experiment = sys.argv[1]
    extra_args = sys.argv[2:]

    script = (_EXPERIMENTS_DIR / experiment).with_suffix(".py")
    if not script.exists():
        print(f"error: '{script}' not found")
        print("\nDid you mean one of:")
        query = experiment.lower()
        for p in _list_experiments():
            rel = p.relative_to(_EXPERIMENTS_DIR).with_suffix("")
            if query in str(rel).lower():
                print(f"  {rel}")
        sys.exit(1)

    result = subprocess.run([sys.executable, str(script)] + extra_args)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

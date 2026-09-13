"""CLI for the Jasper Ridge §6 EDA pipeline (offline, no notebook)."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jasper_eda import run_eda


def main():
    parser = argparse.ArgumentParser(description="Jasper Ridge quantitative EDA and exact FCLS baseline")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Repository root containing data/processed/jasper_ridge")
    args = parser.parse_args()
    result = run_eda(args.root)
    print(f"FCLS status: {result['fcls']['status']}; SLSQP: {result['fcls']['oracle_status']}")


if __name__ == "__main__":
    main()

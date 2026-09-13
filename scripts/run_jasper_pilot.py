"""Run one small contract task, or the complete basic pilot (offline)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jasper_pilot import STAGES, run_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/jasper_pilot.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/jasper_pilot"))
    parser.add_argument("--stage", choices=(*STAGES, "all"), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for stage in STAGES if args.stage == "all" else (args.stage,):
        print(f"Starting {stage}", flush=True)
        run_stage(root, config, output, stage)
        print(f"Completed {stage}: {output}", flush=True)


if __name__ == "__main__":
    main()

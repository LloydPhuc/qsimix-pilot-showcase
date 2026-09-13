"""Prepare Jasper Ridge without importing the project's optional quantum stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def main() -> None:
    code_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=code_root,
                        help="Dataset root; defaults to this script's repository")
    parser.add_argument("--config", type=Path,
                        help="Absolute path, or path relative to --root")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    config_path = args.config or Path("configs/jasper_preprocess.json")
    if not config_path.is_absolute():
        config_path = root / config_path
    sys.path.insert(0, str(code_root / "src"))
    from jasper_preprocess import load_config, prepare_dataset
    config = load_config(config_path)
    manifest = prepare_dataset(root, config)
    print(json.dumps({"output": str(root / "data/processed/jasper_ridge"),
                      "counts": manifest["splits"]["counts"],
                      "orientation": manifest["orientation"],
                      "raw_unchanged": manifest["raw_integrity"]["unchanged"],
                      "warnings": manifest["warnings"]},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

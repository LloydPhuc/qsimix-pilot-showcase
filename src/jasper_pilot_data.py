"""Fixed small subset of the existing Jasper artifact; never rewrites clean data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def select_subset(clean, config):
    y, m = clean["Y"], clean["M"]
    if y.ndim != 2 or m.shape != (y.shape[0], 4):
        raise ValueError("Expected Jasper Y[B,N] and M[B,4]")
    if not 4 <= config["bands"] <= y.shape[0]:
        raise ValueError("bands must be between 4 and the source band count")
    bands = np.linspace(0, y.shape[0] - 1, config["bands"], dtype=int)
    rng = np.random.default_rng(config["seed"])
    chosen = {}
    all_source_ids = []
    for split in ("train", "val", "test"):
        pool = clean[f"{split}_idx"]
        count = config["samples"][split]
        if count <= 0 or count > pool.size or not clean["valid_pixel_mask"][pool].all():
            raise ValueError(f"Invalid sample count or invalid pixels in {split}")
        chosen[split] = np.sort(rng.choice(pool, count, replace=False))
        all_source_ids.extend(pool.tolist())
    if len(set(all_source_ids)) != len(all_source_ids):
        raise ValueError("Source splits overlap")
    # Fit only on original valid train, before subsampling, across all retained bands.
    scale = max(1.0, float(y[:, clean["train_idx"]].max()), float(m.max()))
    if not np.isfinite(scale):
        raise ValueError("Nonfinite training scale")
    ids = np.concatenate(list(chosen.values()))
    a = clean["A_ref"][:, ids]
    if (not np.isfinite(y[:, ids]).all() or np.any(y[:, ids] < 0)
            or not np.isfinite(m).all() or np.any(m < 0)
            or not np.isfinite(a).all() or a.min() < -1e-10
            or np.max(abs(a.sum(axis=0) - 1)) > 1e-8):
        raise ValueError("Nonfinite/negative spectra or invalid reference simplex")
    result = {"Y": y[np.ix_(bands, ids)] / scale, "M": m[bands] / scale,
              "A_ref": a, "pixel_ids": ids, "row": clean["row"][ids],
              "col": clean["col"][ids], "band_indices_clean": bands,
              "band_ids_original": clean["band_ids_original"][bands],
              "material_names": clean["material_names"], "scale": np.array(scale)}
    offset = 0
    for split, selected in chosen.items():
        result[f"{split}_idx"] = np.arange(offset, offset + selected.size)
        offset += selected.size
    return result


def prepare_subset(root, config, output):
    source = root / "data/processed/jasper_ridge/jasper_clean.npz"
    manifest = source.with_name("manifest.json")
    before = sha256(source)
    data = select_subset(load_npz(source), config)
    coverage = {}
    for split in ("train", "val", "test"):
        a = data["A_ref"][:, data[f"{split}_idx"]]
        coverage[split] = {"pixels": a.shape[1], "mixtures_below_095": int((a.max(axis=0) < .95).sum()),
                           "dominant_reference_counts": np.bincount(a.argmax(axis=0), minlength=4).tolist()}
    if coverage["train"]["mixtures_below_095"] == 0:
        raise ValueError("Pure-only calibration cannot learn anchored residual")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "subset.npz", **data)
    if sha256(source) != before:
        raise RuntimeError("Clean input changed during subset creation")
    metadata = {"schema_version": "jasper-pilot-subset-1", "config": config,
                "source_npz_sha256": before, "source_manifest_sha256": sha256(manifest),
                "subset_sha256": sha256(output / "subset.npz"),
                "sampling": "seeded uniform without replacement within existing spatial splits; no label selection",
                "band_selection": "32 by default, linspace integer positions in retained bands; no fitted scoring",
                "scale": float(data["scale"]),
                "scale_fit": "max(1, all original valid train Y, known M); Y/c, M/c, lambda_nominal/c; pi*M/c phase",
                "pixels_outside_model_0_1": int(np.any((data["Y"] < 0) | (data["Y"] > 1), axis=0).sum()),
                "coverage": coverage,
                "array_contract": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in data.items()},
                "scope": "exploratory subsample; reference estimates, not measured ground truth"}
    write_json(output / "subset_manifest.json", metadata)
    return metadata

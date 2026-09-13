"""Conservative Jasper Ridge preprocessing; NumPy/SciPy only.

The supplied M and A are benchmark references. Neither is fitted here. The
predeclared spatial split is exploratory; EDA may describe all scene pixels.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.io import loadmat

RAW_PATHS = ("data/jasperRidge2_R198.mat", "data/Jasper_GT.mat",
             "data/end4_Abundance.fig", "data/end4_Materials.fig")
MATERIAL_NAMES = ("tree", "water", "dirt", "road")
SPLIT_COLUMNS = {"train": [[0, 57]], "val": [[60, 77]], "test": [[80, 99]],
                 "buffer": [[58, 59], [78, 79]]}
SCHEMA_VERSION = "jasper-preprocess-1.0"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_constant(value: str):
    raise ValueError(f"Nonstandard JSON constant: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_config(path: Path) -> dict:
    """Read strict JSON: reject NaN, Infinity and duplicate keys."""
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=_reject_constant,
                      object_pairs_hook=_unique_object)


def _fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _scalar(data: dict, key: str, expected: int) -> int:
    _require(key in data, f"Missing metadata: {key}")
    value = np.asarray(data[key])
    _require(value.shape == (1, 1) and value.dtype.kind in "ui",
             f"{key}: expected integer MATLAB scalar, got {value.shape}/{value.dtype}")
    _require(value.item() == expected, f"{key}: expected {expected}, got {value.item()}")
    return int(value.item())


def _matrix(data: dict, key: str, shape: tuple, dtype: str) -> np.ndarray:
    _require(key in data, f"Missing field: {key}")
    value = np.asarray(data[key])
    _require(value.shape == shape and value.dtype == np.dtype(dtype),
             f"{key}: expected {shape}/{dtype}, got {value.shape}/{value.dtype}")
    return value


def _summary(value: np.ndarray) -> dict:
    finite = value[np.isfinite(value)]
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "range_status": "defined" if finite.size else "undefined_no_finite_values",
            "nonfinite_entries": int((~np.isfinite(value)).sum()),
            "negative_entries": int((value < 0).sum()),
            "zero_entries": int((value == 0).sum())}


def invalid_reason_masks(y: np.ndarray) -> dict[str, np.ndarray]:
    """Fixed policy on B x N spectra; individual zero bands remain valid."""
    _require(y.ndim == 2, "Expected B x N spectra")
    return {"nonfinite_spectrum": ~np.isfinite(y).all(axis=0),
            "negative_spectrum": (y < 0).any(axis=0),
            "all_zero_spectrum": (y == 0).all(axis=0)}


def spatial_indices(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MATLAB column-major pixel IDs; supports non-square verification fixtures."""
    _require(height > 0 and width > 0, "Spatial dimensions must be positive")
    ids = np.arange(height * width, dtype=np.int64)
    return ids, ids % height, ids // height


def _validate_config(config: dict) -> None:
    _require(isinstance(config, dict), "Config must be a JSON object")
    json.dumps(config, allow_nan=False)
    fixed = {"schema_version": SCHEMA_VERSION, "nonnegative_tolerance": 1e-10,
             "abundance_sum_tolerance": 1e-8, "scale_divisor": 5000,
             "seed": 20260913, "split_columns_inclusive": SPLIT_COLUMNS}
    for key, expected in fixed.items():
        _require(config.get(key) == expected, f"Config {key} must match predeclared protocol: {expected}")
    hashes = config.get("expected_raw_sha256", {})
    _require(set(hashes) == set(RAW_PATHS), "Expected SHA256 for all four raw files")
    for value in hashes.values():
        _require(isinstance(value, str) and len(value) == 64
                 and all(c in "0123456789abcdef" for c in value), "Invalid SHA256")
    orientation = config.get("orientation", {})
    _require(orientation.get("order") == "F", "Only the declared MATLAB F mapping is supported")
    _require(orientation.get("status") in ("provisional", "verified"), "Invalid orientation status")
    _require(isinstance(orientation.get("evidence"), str) and bool(orientation["evidence"].strip()),
             "Orientation evidence or provisional caveat is required")
    _require(isinstance(config.get("eda_thresholds_predeclared"), dict), "Missing predeclared EDA thresholds")


def prepare_dataset(root: Path, config: dict) -> dict:
    """Validate raw inputs, export exact contract NPZ, and return strict manifest.

    Paths resolve relative to root, independent of cwd. Raw hashes are pinned
    by config and checked again after export. Arrays are deterministic; no RNG
    is used to assign splits. Failures do not intentionally modify raw data.
    """
    _validate_config(config)
    root = Path(root).expanduser().resolve()
    before = {name: _fingerprint(root / name) for name in RAW_PATHS}
    for name, info in before.items():
        _require(info["sha256"] == config["expected_raw_sha256"][name],
                 f"Raw SHA256 mismatch: {name}")
    source = loadmat(root / RAW_PATHS[0])
    reference = loadmat(root / RAW_PATHS[1])
    y_raw = _matrix(source, "Y", (198, 10000), "uint16")
    m_raw = _matrix(reference, "M", (198, 4), "float64")
    a_raw = _matrix(reference, "A", (4, 10000), "float64")
    metadata = {key: _scalar(source, key, value) for key, value in
                (("nRow", 100), ("nCol", 100), ("nBand", 224), ("maxValue", 5000))}
    bands_raw = _matrix(source, "SlectBands", (198, 1), "uint8")
    bands = bands_raw.ravel().astype(np.int64)
    removed = set(range(1, 4)) | set(range(108, 113)) | set(range(154, 167)) | set(range(220, 225))
    expected_bands = np.array([i for i in range(1, 225) if i not in removed], dtype=np.int64)
    _require(np.array_equal(bands, expected_bands), "Exact retained band IDs mismatch")
    metadata["SlectBands"] = bands.tolist()
    _require("Region" in source, "Missing Region metadata")
    region = source["Region"]
    _require(region.shape == (1, 1) and region.dtype.names is not None,
             "Region must be a scalar MATLAB struct")
    expected_region = {"xStart": 269, "xEnd": 368, "yStart": 105, "yEnd": 204}
    _require(set(region.dtype.names) == set(expected_region), "Unexpected Region fields")
    metadata["Region"] = {key: _scalar({key: region[key][0, 0]}, key, value)
                          for key, value in expected_region.items()}
    _require("cood" in reference, "Missing cood labels")
    cood = reference["cood"]
    _require(cood.shape == (4, 1) and cood.dtype.kind == "O", "Unexpected cood cell array")
    labels = []
    for cell in cood[:, 0]:
        text = np.asarray(cell)
        _require(text.size == 1 and text.dtype.kind == "U", "Unexpected cood string")
        labels.append(str(text.item()))
    _require(labels == [f"{i + 1}-{name}" for i, name in enumerate(MATERIAL_NAMES)],
             f"cood material order mismatch: {labels}")
    metadata["cood"] = labels
    metadata["mat_headers"] = {name: data["__header__"].decode("ascii", errors="replace")
                              for name, data in zip(RAW_PATHS[:2], (source, reference))}
    tolerance = config["nonnegative_tolerance"]
    for name, value in (("M", m_raw), ("A", a_raw)):
        _require(np.isfinite(value).all(), f"{name} contains nonfinite reference values")
        _require(np.all(value >= -tolerance), f"{name} violates nonnegativity tolerance")
    asc_error = float(np.max(np.abs(a_raw.sum(axis=0) - 1.0)))
    _require(asc_error <= config["abundance_sum_tolerance"], "Reference A violates ASC tolerance")
    _require(np.all(a_raw <= 1 + tolerance), "Reference A exceeds upper bound tolerance")
    y = y_raw.astype(np.float64) / metadata["maxValue"]
    m, a = m_raw.copy(), a_raw.copy()
    reasons = invalid_reason_masks(y)
    valid = ~np.logical_or.reduce(list(reasons.values()))
    pixel_ids, row, col = spatial_indices(metadata["nRow"], metadata["nCol"])
    split_labels = np.full(pixel_ids.size, 4, dtype=np.int64)
    splits = {}
    for label, (name, intervals) in enumerate(SPLIT_COLUMNS.items()):
        selected = np.zeros(pixel_ids.size, dtype=bool)
        for start, end in intervals:
            selected |= (col >= start) & (col <= end)
        selected &= valid
        split_labels[selected] = label
        splits[name + "_idx"] = pixel_ids[selected]
    _require(np.array_equal(split_labels != 4, valid), "Split coverage mismatch")
    cube = y.T.reshape(100, 100, 198, order="F")
    _require(np.array_equal(cube.reshape(10000, 198, order="F").T, y, equal_nan=True),
             "Cube roundtrip failed")
    _require(np.array_equal(cube[row, col], y.T, equal_nan=True), "Indexed cube mapping failed")
    recovery_error = float(np.max(np.abs(y * metadata["maxValue"] - y_raw.astype(np.float64))))
    _require(recovery_error <= np.finfo(np.float64).eps * 2 * max(1, int(y_raw.max())),
             "Raw scale recovery exceeds floating point tolerance")
    arrays = {"Y": y, "M": m, "A_ref": a, "band_ids_original": bands,
              "pixel_ids": pixel_ids, "row": row, "col": col,
              "valid_pixel_mask": valid, **splits, "split_labels": split_labels,
              "material_names": np.asarray(MATERIAL_NAMES, dtype="U5"),
              "height": np.asarray(100, dtype=np.int64), "width": np.asarray(100, dtype=np.int64)}
    _, duplicate_counts = np.unique(y_raw.T, axis=0, return_counts=True)
    qc = {"Y_raw": _summary(y_raw), "Y": _summary(y), "M": _summary(m), "A_ref": _summary(a),
          "all_zero_band_ids_original": bands[np.all(y_raw == 0, axis=1)].tolist(),
          "constant_band_ids_original": bands[np.all(y_raw == y_raw[:, :1], axis=1)].tolist(),
          "duplicate_spectrum_groups": int((duplicate_counts > 1).sum()),
          "duplicate_pixels_beyond_first": int(np.maximum(duplicate_counts - 1, 0).sum()),
          "duplicate_pixels_in_groups": int(duplicate_counts[duplicate_counts > 1].sum()),
          "Y_above_one_entries": int((y > 1).sum()),
          "Y_above_one_pixels": int(np.any(y > 1, axis=0).sum()),
          "A_above_one_entries_tolerated": int((a > 1).sum()),
          "A_max_abs_sum_error": asc_error, "reference_repairs": [],
          "raw_scale_recovery_max_abs_error": recovery_error}
    warnings = []
    if config["orientation"]["status"] == "provisional":
        warnings.append("Y F-order and Y-A pixel pairing remain exploratory conventions; FIG verifies only A F-order and M values.")
    warnings.append("Y/5000 is a nominal metadata-based convention, not confirmed physical calibration or common Y-M scale. M-Y band alignment lacks independent wavelength evidence.")
    if qc["Y_above_one_entries"]:
        warnings.append("Y exceeds pilot [0,1] domain; adapter or declared domain relaxation required before training. No clipping.")
    if qc["A_above_one_entries_tolerated"] or (a < 0).any() or (m < 0).any():
        warnings.append("Reference roundoff within tolerance retained unchanged; no projection/repair.")
    output_dir = root / "data/processed/jasper_ridge"
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "jasper_clean.npz"
    np.savez_compressed(npz_path, **arrays)
    with np.load(npz_path, allow_pickle=False) as saved:
        _require(set(saved.files) == set(arrays), "Export key mismatch")
        for key, value in arrays.items():
            loaded = saved[key]
            equal = (np.array_equal(loaded, value, equal_nan=True) if value.dtype.kind == "f"
                     else np.array_equal(loaded, value))
            _require(loaded.dtype == value.dtype and equal, f"Export roundtrip failed: {key}")
    after = {name: _fingerprint(root / name) for name in RAW_PATHS}
    _require(before == after, "Raw input changed during preprocessing; outputs must not be consumed")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "inputs": [{"path": name, **info} for name, info in before.items()],
        "raw_integrity": {"unchanged": True, "expected_sha256_checked": True, "after": after},
        "metadata_original": metadata,
        "source": {"upstream_status": "UNKNOWN", "upstream": "UNKNOWN",
                   "atmospheric_correction": "UNKNOWN", "measurement_units": "UNKNOWN",
                   "license": "UNKNOWN", "wavelength_centers": None,
                   "wavelength_status": "UNKNOWN; use original band IDs",
                   "known_M_protocol": "supplied benchmark reference; not independent training-only extracted M",
                   "GT_status": "reference estimates, not field measurement",
                   "pixel_alignment": "Y columns paired with A_ref columns by benchmark convention; not independently verified",
                   "band_alignment": "M-Y alignment assumed from dimensions and benchmark convention; independent wavelength metadata absent",
                   "use_limit": "exploratory only; provenance insufficient for confirmatory model-effectiveness claims"},
        "scale": {"formula": "Y = Y_raw.astype(float64) / maxValue; M = M_raw; A_ref = A_raw",
                  "divisor": 5000, "status": "nominal_metadata_exploratory_convention", "clipped": False,
                  "evidence": "MAT contains maxValue=5000 but no divisor formula; FIG verifies existing M values only",
                  "physical_calibration_status": "UNVERIFIED",
                  "common_Y_M_scale_status": "UNVERIFIED",
                  "selection": "fixed source maxValue; no residual minimization or test-based scale selection"},
        "orientation": {**config["orientation"], "row_formula": "pixel_id % height",
                        "col_formula": "pixel_id // height",
                        "cube_formula": "Y.T.reshape(height, width, 198, order='F')",
                        "index_base": 0, "roundtrip_verified": True},
        "band_ids_original": bands.tolist(), "material_names": list(MATERIAL_NAMES),
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
        "masks": {"policy": "nonfinite OR negative OR all-zero spectrum; individual zeros retained",
                  "valid_count": int(valid.sum()), "invalid_count": int((~valid).sum()),
                  "reason_mask_encoding": "pixel_ids_true: all other pixel IDs are false; reasons may overlap",
                  "reasons": {key: {"count": int(mask.sum()), "pixel_ids_true": pixel_ids[mask].tolist()}
                              for key, mask in reasons.items()}},
        "splits": {"seed": config["seed"], "seed_usage": "sampling/plots only; split assignment deterministic",
                   "columns_inclusive_zero_based": SPLIT_COLUMNS,
                   "label_codes": {"train": 0, "val": 1, "test": 2, "buffer": 3, "invalid": 4},
                   "counts": {name: int((split_labels == label).sum()) for label, name in
                              enumerate(("train", "val", "test", "buffer", "invalid"))},
                   "protocol": "exploratory/predeclared; test may be observed in EDA",
                   "fit_policy": "future fitted transformations use valid train only; A_ref does not select splits",
                   "patch_policy": "pixelwise baseline; future patches must be checked against two-column buffers"},
        "quality": qc,
        "decisions": {"removed_pixels": 0, "removed_additional_bands": 0, "crop_reapplied": False,
                      "M_unchanged": True, "A_ref_unchanged": True,
                      "duplicates_and_outliers": "retain; flag only", "denoising_or_normalization": "none",
                      "thresholds_fixed_before_EDA": config["eda_thresholds_predeclared"]},
        "warnings": warnings, "config": config,
        "dependencies": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "outputs": {"jasper_clean.npz": {"path": "data/processed/jasper_ridge/jasper_clean.npz", **_fingerprint(npz_path)}}}
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (output_dir / "manifest.json").write_text(payload, encoding="utf-8")
    return manifest

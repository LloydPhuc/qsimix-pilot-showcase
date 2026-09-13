"""Reproducible, descriptive Jasper Ridge EDA; canonical arrays are [band,pixel].

Public APIs: run_eda(root), fcls_exact(Y, M), solve_fcls(Y, M), fit_train_pca.
The inverse solver accepts only spectra and endmembers, never reference labels.
All fitted EDA parameters (PCA and robust flags) use valid train pixels only.
"""
from __future__ import annotations

import hashlib
import html
import itertools
import json
import os
import platform
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "tmp" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.stats import spearmanr

SEED = 20260913
MATERIALS = ("tree", "water", "dirt", "road")
COLORS = ("#27844b", "#287cc1", "#b78143", "#735f8e")
SPLITS = ("train", "val", "test", "buffer")
QUANTILES = (0, .01, .05, .25, .5, .75, .95, .99, 1)


def _json_safe(value):
    """Undefined floats become null; callers supply the accompanying status."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _describe(x):
    x = np.asarray(x)
    finite = x[np.isfinite(x)]
    result = {"n": int(x.size), "finite_count": int(finite.size),
              "nonfinite_count": int(x.size - finite.size),
              "status": "defined" if finite.size else "undefined_no_finite_values"}
    if finite.size:
        result.update(mean=float(finite.mean()), std=float(finite.std()),
                      min=float(finite.min()), max=float(finite.max()),
                      quantiles={str(q): float(v) for q, v in
                                 zip(QUANTILES, np.quantile(finite, QUANTILES))})
    else:
        result.update(mean=None, std=None, min=None, max=None, quantiles=None)
    return result


def simplex_tangent_basis(e: int) -> np.ndarray:
    """Deterministic orthonormal Helmert basis Q, 1.T Q=0."""
    q = np.zeros((e, max(0, e - 1)), dtype=np.float64)
    for j in range(e - 1):
        den = np.sqrt((j + 1) * (j + 2))
        q[:j + 1, j] = 1 / den
        q[j + 1, j] = -(j + 1) / den
    return q


def fcls_exact(Y: np.ndarray, M: np.ndarray, feasibility_tol: float = 1e-10):
    """Return (A[E,N], diagnostics), exact convex FCLS by nonempty faces.

    Each face uses a=a0+Qz, z=lstsq(M_face Q, Y-M_face a0), thereby
    enforcing sum(a)=1 without normal equations. Singular faces are allowed:
    an infeasible minimum-norm representative can be skipped because the same
    optimal spectrum has a representation on a lower-dimensional face. Every
    vertex is visited. Only <= feasibility_tol negative roundoff is repaired.
    Global optimum means floating-point face enumeration, not symbolic exactness.
    """
    y, m = np.asarray(Y, dtype=np.float64), np.asarray(M, dtype=np.float64)
    if y.ndim != 2 or m.ndim != 2 or y.shape[0] != m.shape[0]:
        raise ValueError("Expected Y[B,N] and M[B,E]")
    if not np.isfinite(y).all() or not np.isfinite(m).all():
        raise ValueError("FCLS requires finite Y and M")
    e, n = m.shape[1], y.shape[1]
    if not 1 <= e <= 4:
        raise ValueError("Exact enumerator is limited to 1..4 endmembers")
    if not 0 <= feasibility_tol <= 1e-8:
        raise ValueError("feasibility_tol must remain a roundoff-scale tolerance")
    best = np.full(n, np.inf)
    a_best = np.zeros((e, n), dtype=np.float64)
    best_face = np.zeros(n, dtype=np.int64)
    faces, repairs = [], 0
    for k in range(1, e + 1):
        for support in itertools.combinations(range(e), k):
            idx = np.asarray(support)
            ms, a0 = m[:, idx], np.full((k, 1), 1 / k)
            q = simplex_tangent_basis(k)
            if k == 1:
                candidate = np.ones((1, n), dtype=np.float64)
                rank = 0
            else:
                z, _, rank, _ = np.linalg.lstsq(ms @ q, y - ms @ a0, rcond=None)
                candidate = a0 + q @ z
            feasible = candidate.min(axis=0) >= -feasibility_tol
            ids = np.flatnonzero(feasible)
            faces.append({"support_zero_based": list(support),
                          "tangent_rank": int(rank), "feasible_count": int(ids.size)})
            if not ids.size:
                continue
            c = candidate[:, ids].copy()
            repairs += int(np.count_nonzero(c < 0))
            c[c < 0] = 0  # roundoff only, never projected unconstrained LS
            c /= c.sum(axis=0, keepdims=True)
            r = y[:, ids] - ms @ c
            obj = np.einsum("bn,bn->n", r, r)
            wins = obj < best[ids]
            chosen = ids[wins]
            best[chosen] = obj[wins]
            a_best[:, chosen] = 0
            a_best[np.ix_(idx, chosen)] = c[:, wins]
            best_face[chosen] = sum(1 << i for i in support)
    if not np.isfinite(best).all():
        raise RuntimeError("No feasible simplex face found")
    # Objective convention ||Ma-y||^2; g+nu*1-mu=0, mu>=0.
    gradient = 2 * m.T @ (m @ a_best - y)
    active = a_best > 1e-9
    nu = -(gradient * active).sum(axis=0) / active.sum(axis=0)
    mu = gradient + nu[None, :]
    stationarity = np.max(np.where(active, np.abs(mu), 0), axis=0)
    dual_violation = np.maximum(0, -mu.min(axis=0))
    complementarity = np.max(np.abs(a_best * mu), axis=0)
    asc = np.abs(a_best.sum(axis=0) - 1)
    anc = np.maximum(0, -a_best.min(axis=0))
    kkt = np.maximum.reduce([stationarity, dual_violation, complementarity, asc, anc])
    return a_best, {
        "method": "all_nonempty_simplex_faces_float64_equality_LS",
        "face_count": len(faces), "faces": faces,
        "feasibility_tolerance": feasibility_tol, "active_tolerance": 1e-9,
        "roundoff_negative_entries_repaired_across_faces": repairs,
        "objective": best, "support_bitmask": best_face,
        "kkt_per_pixel": kkt, "stationarity_per_pixel": stationarity,
        "dual_violation_per_pixel": dual_violation,
        "complementarity_per_pixel": complementarity,
        "asc_max_abs": float(asc.max()) if n else None,
        "anc_max_violation": float(anc.max()) if n else None,
        "kkt_max": float(kkt.max()) if n else None,
        "kkt_tolerance": 1e-7,
        "status": "passed" if np.all(kkt <= 1e-7) else "kkt_warning",
    }


def solve_fcls(Y: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Convenience array-only FCLS API, with no reference-label argument."""
    return fcls_exact(Y, M)[0]


def fit_train_pca(Y: np.ndarray, train_idx: np.ndarray) -> dict:
    """Fit ONLY Y[:,train_idx]; components are [component,band]."""
    y = np.asarray(Y, dtype=np.float64)
    idx = np.asarray(train_idx, dtype=np.int64)
    if idx.size < 2 or not np.isfinite(y[:, idx]).all():
        raise ValueError("PCA needs at least two finite train spectra")
    mean = y[:, idx].mean(axis=1)
    xc = y[:, idx] - mean[:, None]
    covariance = xc @ xc.T / (idx.size - 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, components = np.maximum(values[order], 0), vectors[:, order].T
    # Resolve arbitrary eigenvector signs reproducibly (degenerate spaces may rotate).
    signs = np.sign(components[np.arange(len(values)), np.abs(components).argmax(axis=1)])
    components *= signs[:, None]
    total = values.sum()
    ratio = values / total if total > 0 else np.zeros_like(values)
    cumulative = np.cumsum(ratio)
    dims = {str(t): int(np.searchsorted(cumulative, t) + 1) if total > 0 else None
            for t in (.95, .99, .999)}
    return {"mean": mean, "components": components, "explained_variance": values,
            "explained_variance_ratio": ratio, "cumulative_explained_variance": cumulative,
            "dimensions": dims, "fit_pixel_ids": idx.copy(),
            "status": "defined" if total > 0 else "undefined_zero_train_variance"}


def spectral_angle(x, y):
    """Columnwise angle in radians, undefined (NaN) for zero/nonfinite norms."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    nx, ny = np.linalg.norm(x, axis=0), np.linalg.norm(y, axis=0)
    valid = (nx > 0) & (ny > 0) & np.isfinite(nx) & np.isfinite(ny)
    angles = np.full(nx.shape, np.nan)
    cosine = np.einsum("bn,bn->n", x[:, valid], y[:, valid]) / (nx[valid] * ny[valid])
    angles[valid] = np.arccos(np.clip(cosine, -1, 1))
    return angles


def _reconstruction_metrics(y, r, sam):
    if y.shape[1] == 0:
        return {"n_pixels": 0, "status": "undefined_empty_subset", "rmse": None,
                "sre_db": None, "sre_status": "undefined_empty_subset", "sam_radians": _describe([])}
    energy = float(np.sum(y * y))
    error = float(np.sum(r * r))
    if energy > 0 and error > 0:
        sre, status = 10 * np.log10(energy / error), "defined"
    else:
        sre, status = None, "undefined_zero_signal_or_error_energy"
    return {"n_pixels": y.shape[1], "status": "defined",
            "rmse": float(np.sqrt(np.mean(r * r))), "signal_energy": energy,
            "residual_energy": error, "sre_db": sre, "sre_status": status,
            "sam_radians": _describe(sam)}


def _geometry(m):
    def spectrum(x):
        s = np.linalg.svd(x, compute_uv=False)
        rank = int(np.linalg.matrix_rank(x))
        full = rank == min(x.shape)
        return {"singular_values": s, "rank": rank,
                "condition_number": float(s[0] / s[-1]) if full and s[-1] > 0 else None,
                "condition_status": "defined" if full and s[-1] > 0 else "undefined_rank_deficient",
                "sigma_min": float(s[-1])}
    q = simplex_tangent_basis(m.shape[1])
    rows = []
    for i, j in itertools.combinations(range(m.shape[1]), 2):
        angle = spectral_angle(m[:, i:i + 1], m[:, j:j + 1])[0]
        rows.append({"material_1": MATERIALS[i], "material_2": MATERIALS[j],
                     "sad_radians": angle, "sad_degrees": np.degrees(angle),
                     "sad_status": "defined" if np.isfinite(angle) else "undefined_zero_norm",
                     "euclidean_distance": float(np.linalg.norm(m[:, i] - m[:, j]))})
    return {"M": spectrum(m), "MQ": spectrum(m @ q), "pairwise": rows}, q


def _robust_threshold(x, train_idx):
    values = np.asarray(x)[train_idx]
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("No valid train values for robust threshold")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0:
        # The predeclared policy makes a zero-MAD threshold undefined.
        return {"median": median, "mad": mad, "multiplier": 6.0, "mad_scale": 1.4826,
                "threshold": None, "status": "undefined_zero_mad",
                "fit_n": int(values.size)}
    return {"median": median, "mad": mad, "multiplier": 6.0, "mad_scale": 1.4826,
            "threshold": median + 6 * 1.4826 * mad, "status": "defined", "fit_n": int(values.size)}


def _correlation(x, y):
    good = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[good], np.asarray(y)[good]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": len(x), "pearson": None, "spearman": None, "status": "undefined_constant_or_small_n"}
    return {"n": len(x), "pearson": float(np.corrcoef(x, y)[0, 1]),
            "spearman": float(spearmanr(x, y).statistic), "status": "descriptive_no_causal_inference"}


def _segmented(ax, bands, values, **kwargs):
    segments = np.split(np.arange(len(bands)), np.flatnonzero(np.diff(bands) != 1) + 1)
    curve_color = kwargs.get("color")
    for i, segment in enumerate(segments):
        options = dict(kwargs)
        if i and "label" in options:
            options.pop("label")
        if curve_color is not None:
            options["color"] = curve_color
        line, = ax.plot(bands[segment], values[segment], **options)
        curve_color = line.get_color()


def _savefig(fig, directory, name):
    fig.savefig(directory / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


class _Report:
    """Write matching Markdown and offline HTML without a notebook/Markdown dependency."""
    def __init__(self):
        self.md = []
        self.html = []

    def heading(self, text, level=2):
        self.md.append(f"{'#' * level} {text}\n")
        self.html.append(f"<h{level}>{html.escape(text)}</h{level}>")

    def paragraph(self, text):
        self.md.append(text + "\n")
        self.html.append(f"<p>{html.escape(text)}</p>")

    def table(self, frame, caption=""):
        if caption:
            self.paragraph(caption)
        frame = frame.copy()
        def fmt(v):
            if isinstance(v, (float, np.floating)):
                return f"{v:.6g}" if np.isfinite(v) else "undefined"
            return str(v)
        columns = list(frame.columns)
        self.md.extend(["| " + " | ".join(columns) + " |",
                        "| " + " | ".join("---" for _ in columns) + " |"])
        for row in frame.itertuples(index=False, name=None):
            self.md.append("| " + " | ".join(fmt(v) for v in row) + " |")
        self.md.append("")
        self.html.append('<div class="table">' + frame.to_html(index=False, float_format=lambda v: f"{v:.6g}",
                                                               na_rep="undefined", escape=True) + "</div>")

    def figure(self, name, caption):
        self.md.append(f"![{caption}](figures/{name})\n\n{caption}\n")
        self.html.append(f'<figure><img src="figures/{html.escape(name)}" alt="{html.escape(caption)}">'
                         f'<figcaption>{html.escape(caption)}</figcaption></figure>')

    def link(self, path, label):
        self.md.append(f"[{label}]({path})\n")
        self.html.append(f'<p><a href="{html.escape(path)}">{html.escape(label)}</a></p>')

    def write(self, directory):
        (directory / "eda_report.md").write_text("\n".join(self.md), encoding="utf-8")
        css = "body{font:16px/1.65 system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 24px;color:#17232e}h1,h2{line-height:1.25}img{max-width:100%;height:auto}figure{margin:24px 0}figcaption{color:#415366}table{border-collapse:collapse;font-size:14px}th,td{border:1px solid #ccd5de;padding:7px 10px;text-align:right}th{background:#edf3f7}.table{overflow-x:auto}a{color:#165ca6}@media print{body{font-size:11px}figure{break-inside:avoid}}"
        (directory / "eda_report.html").write_text(
            '<!doctype html><html lang="vi"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Jasper Ridge — EDA định lượng</title><style>' + css + '</style></head><body>'
            + "\n".join(self.html) + '</body></html>', encoding="utf-8")


def run_eda(root: Path) -> dict:
    """Read agent B's canonical NPZ/manifest and generate all §6 EDA deliverables."""
    root = Path(root).resolve()
    source = root / "data/processed/jasper_ridge"
    target = root / "reports/jasper_ridge"
    figures, tables = target / "figures", target / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8-sig"))
    with np.load(source / "jasper_clean.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    required = {"Y", "M", "A_ref", "band_ids_original", "pixel_ids", "row", "col",
                "valid_pixel_mask", "train_idx", "val_idx", "test_idx", "buffer_idx",
                "split_labels", "material_names", "height", "width"}
    if required - data.keys():
        raise ValueError(f"Missing canonical fields: {sorted(required - data.keys())}")
    y, m, a = (data[k] for k in ("Y", "M", "A_ref"))
    if y.shape != (198, 10000) or m.shape != (198, 4) or a.shape != (4, 10000):
        raise ValueError("Canonical Jasper shapes required; never infer a transpose")
    if any(x.dtype != np.float64 for x in (y, m, a)):
        raise ValueError("Canonical computation arrays must be float64")
    if tuple(data["material_names"].tolist()) != MATERIALS:
        raise ValueError("Local cood order must be tree/water/dirt/road")
    bands = data["band_ids_original"]
    expected_bands = np.array([b for b in range(1, 225) if not
                              (1 <= b <= 3 or 108 <= b <= 112 or 154 <= b <= 166 or 220 <= b <= 224)])
    if not np.array_equal(bands, expected_bands):
        raise ValueError("Unexpected original band IDs")
    n, h, w = y.shape[1], int(data["height"]), int(data["width"])
    row, col = data["row"], data["col"]
    valid = data["valid_pixel_mask"]
    if valid.shape != (n,) or valid.dtype != np.bool_ or h * w != n:
        raise ValueError("Invalid mask/grid contract")
    if not np.array_equal(data["pixel_ids"], np.arange(n)):
        raise ValueError("Pixel IDs must remain canonical source column IDs")
    if not np.array_equal(row, np.arange(n) % h) or not np.array_equal(col, np.arange(n) // h):
        raise ValueError("Unexpected canonical MATLAB-order coordinates")
    if not np.isfinite(m).all() or not np.isfinite(a).all() or a.min() < -1e-10:
        raise ValueError("Invalid M/A reference")
    if m.min() < -1e-10 or np.max(np.abs(a.sum(axis=0) - 1)) > 1e-8:
        raise ValueError("Reference nonnegativity/simplex validation failed")
    expected_valid = np.isfinite(y).all(axis=0) & (y >= 0).all(axis=0) & np.any(y != 0, axis=0)
    if not np.array_equal(valid, expected_valid):
        raise ValueError("Valid mask must follow the predeclared spectrum policy")
    ids = np.flatnonzero(valid)
    groups = {"all_valid_descriptive": ids, **{s: data[s + "_idx"] for s in SPLITS}}
    expected_labels = np.full(n, 3, dtype=np.int64)
    expected_labels[col <= 57] = 0
    expected_labels[(col >= 60) & (col <= 77)] = 1
    expected_labels[col >= 80] = 2
    expected_labels[~valid] = 4
    if not np.array_equal(data["split_labels"], expected_labels):
        raise ValueError("Split labels violate the predeclared spatial protocol")
    for label, split in enumerate(SPLITS):
        if not np.array_equal(np.sort(groups[split]), np.flatnonzero(expected_labels == label)):
            raise ValueError(f"Split index mismatch: {split}")
    train = groups["train"]
    raw = loadmat(root / "data/jasperRidge2_R198.mat")
    raw_y = raw["Y"]
    divisor = float(np.asarray(raw["maxValue"]).squeeze())
    if raw_y.shape != y.shape or not np.allclose(y, raw_y.astype(np.float64) / divisor,
                                                rtol=0, atol=0, equal_nan=True):
        raise ValueError("Clean Y does not match the declared nominal raw/maxValue scale")

    def csv(name, records):
        frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
        frame.to_csv(tables / name, index=False, encoding="utf-8", na_rep="")
        return frame

    def grid(values):
        result = np.full((h, w), np.nan)
        result[row, col] = values
        return result

    def maps(name, panels, ncols=3, figsize=None):
        nr = int(np.ceil(len(panels) / ncols))
        fig, axes = plt.subplots(nr, ncols, figsize=figsize or (5 * ncols, 4.4 * nr),
                                 squeeze=False, layout="constrained")
        for ax, (values, title, cmap, vmin, vmax) in zip(axes.flat, panels):
            im = ax.imshow(grid(values), origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set(title=title, xlabel="Local column (0-based)", ylabel="Local row (0-based)")
            fig.colorbar(im, ax=ax, shrink=.82)
        for ax in list(axes.flat)[len(panels):]:
            ax.set_visible(False)
        _savefig(fig, figures, name)

    yv = y[:, ids]
    _, inverse, duplicate_counts = np.unique(yv.T, axis=0, return_inverse=True, return_counts=True)
    duplicate_group_size = np.zeros(n, dtype=np.int64)
    duplicate_group_size[ids] = duplicate_counts[inverse]
    finite = np.isfinite(y)
    brightness = np.mean(y, axis=0)
    norm = np.linalg.norm(y, axis=0)
    purity = a.max(axis=0)
    # Only the entropy calculation handles tiny negative reference roundoff.
    a_entropy = np.maximum(a, 0)
    log_a = np.zeros_like(a_entropy)
    np.log(a_entropy, out=log_a, where=a_entropy > 0)
    entropy = -np.sum(a_entropy * log_a, axis=0) / np.log(4)
    dominant = a.argmax(axis=0)
    gap1 = np.flatnonzero(np.diff(bands) == 1)
    differences = y[gap1 + 1] - y[gap1]
    spectral_difference_rms = np.sqrt(np.mean(differences * differences, axis=0))
    spatial_sum, spatial_count = np.zeros(n), np.zeros(n, dtype=np.int64)
    neighbor_records = []
    id_grid = np.arange(n).reshape(h, w, order="F")
    for direction, p, q in (("horizontal", id_grid[:, :-1].ravel(), id_grid[:, 1:].ravel()),
                            ("vertical", id_grid[:-1, :].ravel(), id_grid[1:, :].ravel())):
        good = valid[p] & valid[q]
        p, q = p[good], q[good]
        values = np.sqrt(np.mean((y[:, p] - y[:, q]) ** 2, axis=0))
        np.add.at(spatial_sum, p, values)
        np.add.at(spatial_sum, q, values)
        np.add.at(spatial_count, p, 1)
        np.add.at(spatial_count, q, 1)
        neighbor_records.append({"direction": direction, **_describe(values)})
    spatial_neighbor_rmse = np.divide(spatial_sum, spatial_count, out=np.full(n, np.nan), where=spatial_count > 0)

    print("EDA: exact FCLS on valid pixels", flush=True)
    af_valid, solver = fcls_exact(yv, m)
    af = np.full_like(a, np.nan)
    af[:, ids] = af_valid
    r_ref, r_fcls = y - m @ a, y - m @ af
    r_ref[:, ~valid], r_fcls[:, ~valid] = np.nan, np.nan
    rmse_ref = np.sqrt(np.mean(r_ref ** 2, axis=0))
    rmse_fcls = np.sqrt(np.mean(r_fcls ** 2, axis=0))
    sam_ref, sam_fcls = spectral_angle(y, m @ a), spectral_angle(y, m @ af)
    sam_ref[~valid], sam_fcls[~valid] = np.nan, np.nan
    a_error = af - a
    armse_pixel = np.sqrt(np.mean(a_error ** 2, axis=0))
    rng = np.random.default_rng(SEED)
    oracle_ids = np.sort(rng.choice(ids, size=min(128, ids.size), replace=False))
    oracle_records = []
    for pid in oracle_ids:
        yy = y[:, pid]
        objective = lambda z: float(np.sum((m @ z - yy) ** 2))
        jacobian = lambda z: 2 * m.T @ (m @ z - yy)
        res = minimize(objective, np.full(4, .25), jac=jacobian, method="SLSQP",
                       bounds=[(0, 1)] * 4,
                       constraints={"type": "eq", "fun": lambda z: z.sum() - 1,
                                    "jac": lambda z: np.ones_like(z)},
                       options={"ftol": 1e-12, "maxiter": 1000})
        oracle_records.append({"pixel_id": pid, "success": bool(res.success), "status": int(res.status),
                               "message": res.message, "iterations": res.nit,
                               "objective_slsqp": res.fun, "objective_face": objective(af[:, pid]),
                               "objective_face_minus_slsqp": objective(af[:, pid]) - res.fun,
                               "max_abs_abundance_difference": np.max(np.abs(af[:, pid] - res.x)),
                               "slsqp_asc_error": abs(res.x.sum() - 1),
                               "slsqp_anc_violation": max(0, -res.x.min())})
    oracle = csv("fcls_slsqp_comparison.csv", oracle_records)
    oracle_ok = bool(oracle.success.all() and (oracle.objective_face_minus_slsqp <= 1e-8).all()
                     and (oracle.max_abs_abundance_difference <= 1e-5).all())
    print("EDA: train-only PCA and quantitative tables", flush=True)
    pca = fit_train_pca(y, train)
    scores = pca["components"] @ (y - pca["mean"][:, None])
    scores[:, ~valid] = np.nan
    m_scores = pca["components"] @ (m - pca["mean"][:, None])
    geometry, q = _geometry(m)
    pairwise = csv("endmember_pairwise.csv", geometry["pairwise"])
    csv("pca_explained_variance.csv", {"component_1based": np.arange(1, 199),
                                    "train_explained_variance": pca["explained_variance"],
                                    "train_explained_variance_ratio": pca["explained_variance_ratio"],
                                    "train_cumulative_ratio": pca["cumulative_explained_variance"]})

    stats_band = {"band_id_original": bands, "clean_dtype": [str(y.dtype)] * len(bands),
                  "raw_min": raw_y.min(axis=1), "raw_max": raw_y.max(axis=1),
                  "mean_valid": yv.mean(axis=1), "std_valid": yv.std(axis=1),
                  "nonfinite_count_all": (~finite).sum(axis=1), "zero_count_all": (y == 0).sum(axis=1),
                  "negative_count_all": (y < 0).sum(axis=1), "gt1_count_all": (y > 1).sum(axis=1),
                  "constant_valid": np.ptp(yv, axis=1) == 0,
                  "rmse_reference_valid": np.sqrt(np.mean(r_ref[:, ids] ** 2, axis=1)),
                  "bias_reference_valid": np.mean(r_ref[:, ids], axis=1),
                  "rmse_fcls_valid": np.sqrt(np.mean(r_fcls[:, ids] ** 2, axis=1)),
                  "bias_fcls_valid": np.mean(r_fcls[:, ids], axis=1)}
    for quantile, values in zip(QUANTILES, np.quantile(yv, QUANTILES, axis=1)):
        stats_band[f"q{quantile:g}_valid"] = values
    band_table = csv("band_statistics.csv", stats_band)
    diff_table = csv("adjacent_band_difference.csv", {
        "band_id_left": bands[gap1], "band_id_right": bands[gap1 + 1],
        "mean_difference_valid": differences[:, ids].mean(axis=1),
        "std_difference_valid": differences[:, ids].std(axis=1),
        "rms_difference_valid": np.sqrt(np.mean(differences[:, ids] ** 2, axis=1)),
        "median_abs_difference_valid": np.median(np.abs(differences[:, ids]), axis=1)})
    with np.errstate(divide="ignore", invalid="ignore"):
        band_correlation = np.corrcoef(yv)
    csv("band_correlation.csv", pd.DataFrame(band_correlation, columns=[f"band_{b}" for b in bands])
        .assign(band_id_original=bands).set_index("band_id_original").reset_index())

    abundance_rows, mixture_rows, metric_rows, high_purity_rows = [], [], [], []
    for split, ix in groups.items():
        count = len(ix)
        ar = a[:, ix]
        for threshold in (.90, .95, .99):
            pure = purity[ix] >= threshold
            interior = (~pure) & np.all(ar > .01, axis=0)
            boundary = (~pure) & ~interior
            mixture_rows.append({"split": split, "n": count, "pure_threshold": threshold,
                                 "pure_count": int(pure.sum()), "mixed_count": int((~pure).sum()),
                                 "mixed_interior_all_a_gt_0.01": int(interior.sum()),
                                 "mixed_boundary_min_a_le_0.01": int(boundary.sum())})
        for e, material in enumerate(MATERIALS):
            values, error = ar[e], a_error[e, ix]
            abundance_rows.append({"split": split, "material": material, "n": count,
                                   "mean": np.mean(values) if count else np.nan,
                                   "median": np.median(values) if count else np.nan,
                                   "std": np.std(values) if count else np.nan,
                                   "abundance_sum": np.sum(values),
                                   "mass_fraction_reference": np.sum(values) / np.sum(ar) if count else np.nan,
                                   "dominant_count": int(np.sum(dominant[ix] == e)),
                                   **{f"pure_ge_{t:.2f}": int(np.sum(values >= t)) for t in (.90, .95, .99)},
                                   "fcls_armse_vs_reference": np.sqrt(np.mean(error ** 2)) if count else np.nan,
                                   "fcls_bias_vs_reference": np.mean(error) if count else np.nan,
                                   "fcls_mae_vs_reference": np.mean(np.abs(error)) if count else np.nan})
            for threshold in (.90, .95, .99):
                chosen = ix[values >= threshold]
                mean_spectrum = y[:, chosen].mean(axis=1) if len(chosen) else None
                high_purity_rows.append({"split": split, "material": material,
                                        "threshold": threshold, "n_selected_using_A_ref": len(chosen),
                                        "mean_spectrum_rmse_to_M": np.sqrt(np.mean((mean_spectrum - m[:, e]) ** 2)) if len(chosen) else None,
                                        "mean_spectrum_sad_degrees_to_M": np.degrees(spectral_angle(mean_spectrum[:, None], m[:, e:e + 1])[0]) if len(chosen) else None,
                                        "status": "descriptive_A_ref_selection" if len(chosen) else "undefined_empty_selection"})
        for label, residual, angle in (("reference", r_ref, sam_ref), ("fcls", r_fcls, sam_fcls)):
            metrics = _reconstruction_metrics(y[:, ix], residual[:, ix], angle[ix])
            metric_rows.append({"split": split, "model": label,
                                **{k: v for k, v in metrics.items() if k != "sam_radians"},
                                "sam_mean_radians": metrics["sam_radians"]["mean"],
                                "sam_defined_count": metrics["sam_radians"]["finite_count"],
                                "armse_vs_reference": np.sqrt(np.mean(a_error[:, ix] ** 2)) if label == "fcls" and count else None})
    abundance_table = csv("abundance_by_material_split.csv", abundance_rows)
    mixtures = csv("mixture_coverage_by_split.csv", mixture_rows)
    metrics_table = csv("reconstruction_by_split.csv", metric_rows)
    high_purity = csv("high_purity_reference_vs_M.csv", high_purity_rows)

    thresholds = {key: _robust_threshold(values, train) for key, values in
                  (("brightness", brightness), ("reference_rmse", rmse_ref), ("fcls_rmse", rmse_fcls))}
    flags = {key: (valid & (values > thresholds[key]["threshold"]) if thresholds[key]["threshold"] is not None
                   else np.zeros(n, dtype=bool)) for key, values in
             (("brightness", brightness), ("reference_rmse", rmse_ref), ("fcls_rmse", rmse_fcls))}
    kkt_full = np.full(n, np.nan)
    kkt_full[ids] = solver["kkt_per_pixel"]
    pixel_table = csv("pixel_statistics.csv", {
        "pixel_id": data["pixel_ids"], "row": row, "col": col, "valid": valid,
        "split_label": data["split_labels"], "raw_min": raw_y.min(axis=0), "raw_max": raw_y.max(axis=0),
        "scaled_min": y.min(axis=0), "scaled_max": y.max(axis=0), "brightness_mean_band": brightness,
        "spectrum_l2_norm": norm, "zero_band_count": (y == 0).sum(axis=0),
        "nonfinite_band_count": (~finite).sum(axis=0), "negative_band_count": (y < 0).sum(axis=0),
        "gt1_band_count": (y > 1).sum(axis=0), "duplicate_group_size_valid": duplicate_group_size,
        "constant_spectrum": np.ptp(y, axis=0) == 0, "purity_max_A_ref": purity,
        "entropy_normalized_A_ref": entropy, "dominant_reference_material": np.asarray(MATERIALS)[dominant],
        "adjacent_original_band_difference_rms_proxy": spectral_difference_rms,
        "spatial_neighbor_rmse_proxy": spatial_neighbor_rmse,
        "rmse_reference": rmse_ref, "rmse_fcls": rmse_fcls,
        "sam_reference_radians": sam_ref, "sam_fcls_radians": sam_fcls,
        "fcls_armse_vs_reference": armse_pixel, "fcls_kkt_max_violation": kkt_full,
        **{f"flag_{k}": v for k, v in flags.items()},
        **{f"A_fcls_{material}": af[e] for e, material in enumerate(MATERIALS)},
        **{f"A_fcls_minus_reference_{material}": a_error[e] for e, material in enumerate(MATERIALS)}})
    outlier_frames = []
    for ranking, values in (("brightness", brightness), ("reference_rmse", rmse_ref), ("fcls_rmse", rmse_fcls)):
        top = ids[np.lexsort((ids, -values[ids]))[:20]]
        frame = pixel_table.iloc[top].copy()
        frame.insert(0, "ranking", ranking)
        frame.insert(1, "rank", np.arange(1, len(top) + 1))
        frame["retained_in_clean"] = True
        outlier_frames.append(frame)
    outliers = csv("top_outliers_retained.csv", pd.concat(outlier_frames, ignore_index=True))
    flag_rows = [{"split": split, "flag": key, "count": int(value[ix].sum()), "n": len(ix),
                  "fraction": float(value[ix].mean()) if len(ix) else None}
                 for split, ix in groups.items() for key, value in flags.items()]
    flag_table = csv("robust_flags_by_split.csv", flag_rows)
    correlation_rows = []
    for split, ix in groups.items():
        for residual_name, residual in (("reference_rmse", rmse_ref), ("fcls_rmse", rmse_fcls)):
            for feature, values in (("brightness", brightness), ("purity_A_ref", purity),
                                    ("entropy_A_ref", entropy), ("spatial_neighbor_proxy", spatial_neighbor_rmse),
                                    ("adjacent_band_proxy", spectral_difference_rms)):
                correlation_rows.append({"split": split, "residual": residual_name, "feature": feature,
                                         **_correlation(residual[ix], values[ix])})
    correlations = csv("residual_confounding_correlations.csv", correlation_rows)
    # Within-dominant-material views expose some composition confounding.
    stratum_rows = []
    for split, ix in groups.items():
        for e, material in enumerate(MATERIALS):
            chosen = ix[dominant[ix] == e]
            for label, residual in (("reference_rmse", rmse_ref), ("fcls_rmse", rmse_fcls)):
                stratum_rows.append({"split": split, "dominant_reference": material,
                                     "residual": label, "feature": "brightness",
                                     **_correlation(residual[chosen], brightness[chosen])})
    stratified_correlations = csv("residual_brightness_within_dominant_material.csv", stratum_rows)

    inventory = {"statistics_scope": "all_scene_descriptive; valid subsets explicitly named",
                 "raw_Y": {"dtype": str(raw_y.dtype), "shape": list(raw_y.shape), **_describe(raw_y)},
                 "clean_Y": {"dtype": str(y.dtype), "shape": list(y.shape), **_describe(y)},
                 "M": {"dtype": str(m.dtype), "shape": list(m.shape), **_describe(m)},
                 "A_ref": {"dtype": str(a.dtype), "shape": list(a.shape), **_describe(a)},
                 "nominal_divisor": divisor, "nonfinite_entries": int((~finite).sum()),
                 "negative_entries": int((y < 0).sum()), "zero_entries": int((y == 0).sum()),
                 "all_zero_pixels": int(np.all(y == 0, axis=0).sum()),
                 "all_zero_bands": int(np.all(y == 0, axis=1).sum()),
                 "constant_bands_valid": int(np.sum(np.ptp(yv, axis=1) == 0)),
                 "constant_spectra_valid": int(np.sum(np.ptp(yv, axis=0) == 0)),
                 "duplicate_groups_valid": int(np.sum(duplicate_counts > 1)),
                 "duplicate_pixels_in_groups_valid": int(np.sum(duplicate_counts[duplicate_counts > 1])),
                 "duplicate_excess_copies_valid": int(np.sum(duplicate_counts - 1)),
                 "gt1_entries": int(np.sum(y > 1)), "gt1_pixels": int(np.any(y > 1, axis=0).sum()),
                 "valid_pixels": int(valid.sum()), "invalid_pixels": int((~valid).sum()),
                 "reference_asc_max_abs": float(np.max(np.abs(a.sum(axis=0) - 1))),
                 "scale_policy": "Y_raw.astype(float64)/maxValue; M and A_ref unchanged; Y>1 retained"}
    pca_summary = {"fit_scope": "valid train only", "fit_count": len(train),
                   "fit_pixel_ids_sha256": hashlib.sha256(train.astype("<i8").tobytes()).hexdigest(),
                   "status": pca["status"], "dimensions_for_cumulative_variance": pca["dimensions"],
                   "explained_variance": pca["explained_variance"],
                   "explained_variance_ratio": pca["explained_variance_ratio"],
                   "cumulative_explained_variance": pca["cumulative_explained_variance"]}
    solver_summary = {k: v for k, v in solver.items() if not isinstance(v, np.ndarray)}
    solver_summary.update(kkt=_describe(solver["kkt_per_pixel"]),
                          oracle_seed=SEED, oracle_n=len(oracle),
                          oracle_success_count=int(oracle.success.sum()),
                          oracle_max_abs_objective_difference=float(np.abs(oracle.objective_face_minus_slsqp).max()),
                          oracle_max_face_minus_slsqp=float(oracle.objective_face_minus_slsqp.max()),
                          oracle_max_abs_abundance_difference=float(oracle.max_abs_abundance_difference.max()),
                          oracle_status="passed" if oracle_ok else "failed",
                          reference_used_in_solver=False)
    summary = {"schema_version": "jasper_eda_v1", "seed": SEED,
               "scope": "exploratory_predeclared_split; whole-scene statistics descriptive; test observed in EDA",
               "undefined_policy": "JSON nonfinite -> null; status fields explain undefined metrics; CSV empty cells; NPZ NaN marks invalid/undefined",
               "input_npz_sha256": hashlib.sha256((source / "jasper_clean.npz").read_bytes()).hexdigest(),
               "input_manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
               "manifest_schema_version": manifest.get("schema_version"),
               "material_order": MATERIALS, "band_ids_original": bands,
               "split_counts": {k: len(v) for k, v in groups.items()},
               "inventory": inventory, "geometry": geometry, "pca": pca_summary,
               "fcls": solver_summary, "reconstruction_by_split": metric_rows,
               "abundance_by_material_split": abundance_rows, "mixture_coverage_by_split": mixture_rows,
               "high_purity_reference_vs_M": high_purity_rows,
               "robust_thresholds_train": thresholds, "robust_flags_by_split": flag_rows,
               "residual_confounding_correlations": correlation_rows,
               "residual_brightness_within_dominant_material": stratum_rows,
               "proxies": {"physical_snr": False, "adjacent_original_gap1_pair_count": len(gap1),
                           "excluded_gap_pair_count": len(bands) - 1 - len(gap1),
                           "brightness_valid": _describe(brightness[ids]),
                           "adjacent_band_difference_rms_valid": _describe(spectral_difference_rms[ids]),
                           "spatial_neighbor_rmse_valid": _describe(spatial_neighbor_rmse[ids]),
                           "spatial_pairs_by_direction": neighbor_records},
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "pandas": pd.__version__}}
    arrays = {"pixel_ids": data["pixel_ids"], "row": row, "col": col, "valid_pixel_mask": valid,
              "split_labels": data["split_labels"], "band_ids_original": bands,
              "material_names": data["material_names"], "A_fcls": af,
              "A_fcls_minus_A_ref": a_error, "residual_reference": r_ref, "residual_fcls": r_fcls,
              "rmse_reference_per_pixel": rmse_ref, "rmse_fcls_per_pixel": rmse_fcls,
              "sam_reference_radians_per_pixel": sam_ref, "sam_fcls_radians_per_pixel": sam_fcls,
              "armse_fcls_vs_reference_per_pixel": armse_pixel, "fcls_kkt_violation_per_pixel": kkt_full,
              "pca_mean_train": pca["mean"], "pca_components_train": pca["components"],
              "pca_scores_all_pixels": scores, "pca_endmember_scores": m_scores,
              "pca_explained_variance_train": pca["explained_variance"],
              "pca_explained_variance_ratio_train": pca["explained_variance_ratio"],
              "pca_fit_pixel_ids": train, "simplex_tangent_Q": q,
              "brightness_mean_band_per_pixel": brightness, "spectrum_l2_norm_per_pixel": norm,
              "purity_max_A_ref_per_pixel": purity, "entropy_normalized_A_ref_per_pixel": entropy,
              "dominant_reference_index_per_pixel": dominant,
              "adjacent_band_difference_rms_proxy_per_pixel": spectral_difference_rms,
              "spatial_neighbor_rmse_proxy_per_pixel": spatial_neighbor_rmse,
              "band_correlation_all_valid_descriptive": band_correlation,
              "duplicate_group_size_valid_per_pixel": duplicate_group_size,
              "slsqp_comparison_pixel_ids": oracle_ids,
              **{f"flag_{k}_train_threshold": v for k, v in flags.items()}}
    for key in ("support_bitmask", "stationarity_per_pixel", "dual_violation_per_pixel", "complementarity_per_pixel", "objective"):
        full = np.full(n, np.nan)
        full[ids] = solver[key]
        arrays[f"fcls_{key}"] = full
    np.savez_compressed(target / "eda_arrays.npz", **arrays)
    summary["array_contract"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()}

    print("EDA: rendering offline figures", flush=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.grid": False})
    composite_indices = np.array([120, 70, 20])
    limits = np.quantile(y[composite_indices][:, train], [.02, .98], axis=1)
    rgb = np.clip((y[composite_indices] - limits[0, :, None]) /
                  np.maximum(limits[1] - limits[0], np.finfo(float).eps)[:, None], 0, 1)
    rgb[:, ~valid] = 0
    rgb_grid = np.stack([grid(channel) for channel in rgb], axis=-1)
    summary["composite"] = {"retained_band_indices_zero_based_RGB": composite_indices,
                            "original_band_ids_RGB": bands[composite_indices], "display_fit_scope": "train",
                            "display_quantiles": [.02, .98], "display_limits": limits,
                            "status": "arbitrary_band_composite_not_true_color; display clipping only"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    axes[0].imshow(rgb_grid, origin="upper")
    axes[0].set_title(f"Band composite RGB original IDs {bands[composite_indices].tolist()}")
    for ax, values, title in zip(axes[1:], (brightness, norm), ("Mean-band intensity (nominal scale)", "Spectral L2 norm (nominal scale)")):
        im = ax.imshow(grid(values), origin="upper", cmap="viridis")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=.8)
    for ax in axes:
        ax.set(xlabel="Local column (0-based)", ylabel="Local row (0-based)")
    _savefig(fig, figures, "scene_overview.png")
    maps("reference_abundance.png", [(a[e], f"A_ref: {material}", "viridis", 0, 1)
                                      for e, material in enumerate(MATERIALS)], ncols=2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    im = axes[0].imshow(grid(dominant), cmap=ListedColormap(COLORS), vmin=-.5, vmax=3.5)
    cb = fig.colorbar(im, ax=axes[0], ticks=np.arange(4), shrink=.85)
    cb.ax.set_yticklabels(MATERIALS)
    axes[0].set_title("Dominant reference (argmax A_ref; ties -> first)")
    im = axes[1].imshow(grid(data["split_labels"]), cmap=ListedColormap(["#4c9f70", "#e4a536", "#7469b6", "#c5c9ce", "#252525"]), vmin=-.5, vmax=4.5)
    cb = fig.colorbar(im, ax=axes[1], ticks=np.arange(5), shrink=.85)
    cb.ax.set_yticklabels(["train", "val", "test", "buffer", "invalid"])
    axes[1].set_title("Predeclared spatial split")
    for ax in axes:
        ax.set(xlabel="Local column (0-based)", ylabel="Local row (0-based)")
    _savefig(fig, figures, "dominant_and_split.png")
    maps("qc_maps.png", [(~valid, "Invalid mask (1=invalid)", "gray_r", 0, 1),
                         ((y == 0).sum(axis=0), "Zero bands per pixel", "magma", 0, None),
                         ((y > 1).sum(axis=0), "Bands >1 per pixel (retained)", "magma", 0, None),
                         (duplicate_group_size > 1, "Duplicate valid spectrum membership", "gray_r", 0, 1),
                         (flags["brightness"], "High brightness: train robust threshold", "gray_r", 0, 1),
                         (flags["fcls_rmse"], "High FCLS RMSE: train robust threshold", "gray_r", 0, 1)])
    maps("purity_entropy_proxies.png", [(purity, "Purity: max A_ref", "viridis", 0, 1),
                                        (entropy, "Reference entropy / log(4)", "viridis", 0, 1),
                                        (spectral_difference_rms, "Adjacent original-band RMS proxy", "magma", 0, None),
                                        (spatial_neighbor_rmse, "Spatial neighbor RMSE proxy", "magma", 0, None)], ncols=2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    segments = np.split(np.arange(len(bands)), np.flatnonzero(np.diff(bands) != 1) + 1)
    q05, q50, q95 = np.quantile(yv, [.05, .5, .95], axis=1)
    for i, segment in enumerate(segments):
        axes[0, 0].fill_between(bands[segment], q05[segment], q95[segment], alpha=.2, color="#1f77b4",
                              label="5–95%" if i == 0 else None)
    _segmented(axes[0, 0], bands, yv.mean(axis=1), label="Mean")
    _segmented(axes[0, 0], bands, q50, label="Median", linestyle="--")
    _segmented(axes[0, 0], bands, yv.std(axis=1), label="Std", color="#97499b")
    axes[0, 0].set(title=f"All valid scene spectra (n={len(ids)}), descriptive", ylabel="Nominal scaled value")
    axes[0, 0].legend()
    for e, material in enumerate(MATERIALS):
        _segmented(axes[0, 1], bands, m[:, e], label=material, color=COLORS[e])
    axes[0, 1].set(title="Local reference endmembers M (unchanged)", ylabel="Nominal scale")
    axes[0, 1].legend()
    for split in SPLITS:
        axes[1, 0].hist(brightness[groups[split]], bins=45, alpha=.4, density=True, label=f"{split} n={len(groups[split])}")
    axes[1, 0].set(title="Brightness: mean over retained bands", xlabel="Mean-band value", ylabel="Density")
    axes[1, 0].legend()
    # Invalid gap pairs are NaN, so no line crosses a removed band interval.
    diff_plot = np.full(len(bands), np.nan)
    diff_plot[gap1] = diff_table.rms_difference_valid
    axes[1, 1].plot(bands, diff_plot)
    axes[1, 1].set(title=f"Adjacent difference RMS ({len(gap1)} original gap=1 pairs)",
                   ylabel="RMS difference; not physical SNR")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.set_xlabel("Original band ID (1-based); gaps preserved")
    _savefig(fig, figures, "spectral_statistics.png")
    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    # Expand onto original IDs to visibly mask removed bands on both axes.
    corr_original = np.full((224, 224), np.nan)
    corr_original[np.ix_(bands - 1, bands - 1)] = band_correlation
    im = ax.imshow(corr_original, origin="lower", extent=(.5, 224.5, .5, 224.5), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set(title=f"Band correlation: all valid scene n={len(ids)} (white=removed/undefined)",
           xlabel="Original band ID", ylabel="Original band ID")
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    _savefig(fig, figures, "band_correlation.png")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    pure_spectrum_rows = []
    for e, ax in enumerate(axes.flat):
        selected = ids[a[e, ids] >= .95]
        _segmented(ax, bands, m[:, e], color=COLORS[e], label="M reference", linewidth=2)
        if len(selected):
            mean_selected = y[:, selected].mean(axis=1)
            std_selected = y[:, selected].std(axis=1)
            _segmented(ax, bands, mean_selected, color="black", linestyle="--", label="Selected mean")
            for segment in segments:
                ax.fill_between(bands[segment], (mean_selected - std_selected)[segment],
                                (mean_selected + std_selected)[segment], color="grey", alpha=.2)
            for b, mu, sd in zip(bands, mean_selected, std_selected):
                pure_spectrum_rows.append({"material": MATERIALS[e], "threshold": .95,
                                           "n": len(selected), "band_id_original": b,
                                           "mean_selected": mu, "std_selected": sd})
        ax.set(title=f"{MATERIALS[e]}: A_ref >=0.95, all scene n={len(selected)}",
               xlabel="Original band ID; gaps preserved", ylabel="Nominal scale")
        ax.legend()
    csv("high_purity_spectra_threshold_0.95.csv", pure_spectrum_rows)
    _savefig(fig, figures, "high_purity_spectra.png")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    axes[0, 0].plot(np.arange(1, 199), pca["cumulative_explained_variance"], color="#145b91")
    for level in (.95, .99, .999):
        axes[0, 0].axhline(level, linestyle="--", alpha=.5, label=f"{100*level:g}%: {pca['dimensions'][str(level)]} PCs")
    axes[0, 0].set(xlabel="Number of PCs", ylabel="Cumulative train variance", title=f"PCA fit: valid train n={len(train)}", ylim=(0, 1.015))
    axes[0, 0].legend()
    plot_ids = np.sort(rng.choice(ids, min(4000, len(ids)), replace=False))
    for ax, (pcx, pcy) in zip([axes[0, 1], axes[1, 0], axes[1, 1]], [(0, 1), (0, 2), (1, 2)]):
        ax.scatter(scores[pcx, plot_ids], scores[pcy, plot_ids], c=np.asarray(COLORS)[dominant[plot_ids]], s=3, alpha=.2, rasterized=True)
        for e, material in enumerate(MATERIALS):
            ax.scatter(m_scores[pcx, e], m_scores[pcy, e], marker="X", c=COLORS[e], edgecolor="black", s=130, label=material)
        ax.set(xlabel=f"PC{pcx+1}", ylabel=f"PC{pcy+1}", title=f"All-scene display sample n={len(plot_ids)}; X=M")
        ax.legend(fontsize=8)
    _savefig(fig, figures, "pca_train_only.png")
    summary["pca"]["scatter_display_sample_n"] = len(plot_ids)
    common_max = float(max(rmse_ref[ids].max(), rmse_fcls[ids].max()))
    maps("residual_maps.png", [(rmse_ref, "Reference reconstruction RMSE", "magma", 0, common_max),
                               (rmse_fcls, "FCLS reconstruction RMSE", "magma", 0, common_max),
                               (rmse_ref - rmse_fcls, "Reference RMSE minus FCLS RMSE", "viridis", 0, None),
                               (armse_pixel, "FCLS aRMSE versus reference", "magma", 0, None),
                               (sam_ref, "Reference SAM (radians)", "magma", 0, None),
                               (sam_fcls, "FCLS SAM (radians)", "magma", 0, None)])
    limit = float(np.nanmax(np.abs(a_error)))
    maps("fcls_reference_mismatch.png", [(a_error[e], f"FCLS minus A_ref: {material}", "coolwarm", -limit, limit)
                                         for e, material in enumerate(MATERIALS)], ncols=2)
    maps("fcls_abundance.png", [(af[e], f"FCLS inferred: {material}", "viridis", 0, 1)
                              for e, material in enumerate(MATERIALS)], ncols=2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for label, residual, color in (("Reference", r_ref, "#b64936"), ("FCLS", r_fcls, "#276da1")):
        _segmented(axes[0, 0], bands, np.sqrt(np.mean(residual[:, ids] ** 2, axis=1)), label=label, color=color)
        _segmented(axes[0, 1], bands, residual[:, ids].mean(axis=1), label=label, color=color)
        axes[1, 0].hist(np.sqrt(np.mean(residual[:, ids] ** 2, axis=0)), bins=70, alpha=.5, label=label, color=color)
    for ax in axes[0]:
        ax.set_xlabel("Original band ID; gaps preserved")
        ax.legend()
    axes[0, 0].set(title="Per-band reconstruction RMSE, all valid scene", ylabel="Nominal scale")
    axes[0, 1].set(title="Signed residual mean: Y - reconstruction", ylabel="Nominal scale")
    axes[1, 0].set(title="Per-pixel reconstruction RMSE", xlabel="RMSE", ylabel="Pixel count")
    axes[1, 0].legend()
    top3 = ids[np.lexsort((ids, -rmse_fcls[ids]))[:3]]
    for pid in top3:
        _segmented(axes[1, 1], bands, r_fcls[:, pid], label=f"id={pid} (r={row[pid]}, c={col[pid]})")
    axes[1, 1].set(title="Top 3 FCLS residual spectra (pixels retained)", xlabel="Original band ID", ylabel="Y - M A_fcls")
    axes[1, 1].legend(fontsize=8)
    _savefig(fig, figures, "residual_spectra.png")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    for i, (label, residual) in enumerate((("Reference", rmse_ref), ("FCLS", rmse_fcls))):
        for ax, (feature, values) in zip(axes[i], (("Mean-band brightness", brightness), ("Reference purity", purity), ("Reference entropy", entropy))):
            ax.scatter(values[plot_ids], residual[plot_ids], s=3, alpha=.25, c=np.asarray(COLORS)[dominant[plot_ids]], rasterized=True)
            ax.set(xlabel=feature, ylabel=f"{label} RMSE", title=f"Descriptive sample n={len(plot_ids)}")
    _savefig(fig, figures, "residual_confounding.png")

    print("EDA: writing quantitative Vietnamese report", flush=True)
    _write_report(target, summary, band_table, abundance_table, mixtures, metrics_table,
                  high_purity, pairwise, oracle, flag_table, correlations, stratified_correlations, outliers)
    safe = _json_safe(summary)
    (target / "eda_summary.json").write_text(json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if solver["status"] != "passed" or not oracle_ok:
        raise RuntimeError("FCLS validation failed; diagnostic artifacts were written, inspect before use")
    print(f"EDA complete: {target}", flush=True)
    return safe


def _write_report(target, s, bands, abundance, mixtures, metrics, high_purity, pairs,
                  oracle, flags, correlations, strata, outliers):
    r = _Report()
    inv, geom, pc, solver = s["inventory"], s["geometry"], s["pca"], s["fcls"]
    all_metrics = metrics[metrics.split == "all_valid_descriptive"].set_index("model")
    ref, fcls = all_metrics.loc["reference"], all_metrics.loc["fcls"]
    overall = abundance[abundance.split == "all_valid_descriptive"]
    train_ab = abundance[abundance.split == "train"]
    r.heading("Jasper Ridge — báo cáo EDA định lượng và chẩn đoán LMM", 1)
    r.paragraph(f"Phạm vi: toàn cảnh {inv['clean_Y']['shape'][1]:,} pixel × 198 band, "
                f"{inv['valid_pixels']:,} pixel hợp lệ; thứ tự vật liệu local cood là tree/water/dirt/road. "
                f"FCLS đạt RMSE phổ {fcls.rmse:.8f}, aRMSE {fcls.armse_vs_reference:.8f} so với reference; "
                f"LMM dùng A_ref có RMSE {ref.rmse:.8f}. Đây là baseline known-M và mô tả dữ liệu, chưa huấn luyện QSiMix.")
    r.heading("1. Phạm vi bằng chứng, nguồn và protocol")
    r.paragraph("Mọi thống kê toàn cảnh, correlation, bản đồ, phổ high-purity và phân tích test trong báo cáo đều là mô tả exploratory. "
                "Spatial split đã khai báo trước; vì test đã được xem trong EDA, không gọi nó là test confirmatory hoàn toàn chưa quan sát. "
                "PCA, ngưỡng median/MAD và khoảng stretch composite chỉ fit bằng valid train. FCLS là suy luận từng pixel dùng Y,M cố định, "
                "không dùng A_ref để khởi tạo, chọn nghiệm hoặc hiệu chỉnh scale. M là reference được cung cấp; chưa chứng minh nó được thu độc lập khỏi toàn cảnh.")
    r.paragraph("Zhu 2017 §IV-B mô tả abundance labeling bằng constrained least squares hoặc thuật toán với priors bổ sung. "
                "Điều này không chứng minh đầy đủ rằng mọi cột A local chính là output của quy trình đó với đúng M và scale hiện tại. "
                "File tên GT không biến A_ref thành đo thực địa. Chưa đủ bằng chứng để khẳng định reflectance đo tuyệt đối, atmospheric correction, "
                "nguồn GitHub chính xác hoặc giấy phép của bản local. Báo cáo dùng thang nominal raw/maxValue; các giới hạn nguồn cần đọc cùng source audit.")
    r.link("source_audit.md", "Source audit do agent A phụ trách")
    r.link("research_decisions.md", "Đối chiếu PDF với metadata và nguồn sơ cấp")
    r.link("https://ar5iv.labs.arxiv.org/html/1708.05125", "Zhu 2017: phương pháp tạo nhãn tham chiếu, §IV")
    r.link("qa_review.md", "Independent QA do agent D phụ trách")
    r.link("../../data/processed/jasper_ridge/manifest.json", "Manifest preprocessing và bằng chứng orientation")
    r.table(pd.DataFrame([{"split": k, "n_pixels": v} for k, v in s["split_counts"].items()]),
            "Split theo cột local: train 0–57; buffer 58–59; val 60–77; buffer 78–79; test 80–99. Invalid có label=4 và không fit.")
    r.paragraph("Pixel ID là chỉ số cột Y gốc, 0-based; row=id%100, col=id//100, origin ảnh ở trên. "
                "FIG xác nhận F-order của A và giá trị M chính xác; chưa chứng minh độc lập đăng ký pixel Y–A, band M–Y hoặc thang chung Y–M. "
                "F-order của Y và phép chia 5000 là convention exploratory dựa trên metadata/mã tham khảo. "
                "Các metric dùng chung Y,M,A có điều kiện theo các convention này. Đối chiếu bản đồ cho tính nhất quán trực quan, không thay thế provenance. "
                "Roundtrip reshape chỉ xác nhận tính nhất quán số học. Buffer chưa tham gia fit. "
                "Nếu nghiên cứu sau dùng patches, cần kiểm footprint và buffer trước khi sử dụng protocol này.")
    r.heading("2. Inventory, scale và numerical QC")
    inventory_rows = []
    for key in ("raw_Y", "clean_Y", "M", "A_ref"):
        d = inv[key]
        inventory_rows.append({"array": key, "shape": str(d["shape"]), "dtype": d["dtype"],
                               "min": d["min"], "max": d["max"], "mean": d["mean"], "std": d["std"],
                               "nonfinite": d["nonfinite_count"]})
    r.table(pd.DataFrame(inventory_rows))
    r.paragraph(f"Y được cast float64 rồi chia nominal maxValue={inv['nominal_divisor']:g}; M không chia lần nữa, A_ref không đổi. "
                f"Có {inv['gt1_entries']:,} giá trị Y>1 trên {inv['gt1_pixels']:,} pixel, max={inv['clean_Y']['max']:.6g}; tất cả giữ nguyên. "
                "Không clip clean, không chuẩn hóa từng pixel, không denoise, không bỏ band/crop lần hai. Metadata nBand=224 là số band gốc; "
                "198 hàng hiện tại đã bỏ các khoảng 1–3,108–112,154–166,220–224.")
    r.table(pd.DataFrame([{"QC": k, "count_or_error": inv[k]} for k in
                          ("nonfinite_entries", "negative_entries", "zero_entries", "all_zero_pixels", "all_zero_bands",
                           "constant_bands_valid", "constant_spectra_valid", "duplicate_groups_valid",
                           "duplicate_pixels_in_groups_valid", "duplicate_excess_copies_valid", "invalid_pixels",
                           "reference_asc_max_abs")]))
    r.table(pd.DataFrame([{"quantile": k, "raw_Y": inv["raw_Y"]["quantiles"][k],
                          "scaled_Y": inv["clean_Y"]["quantiles"][k]} for k in inv["clean_Y"]["quantiles"]]),
            "Quantile trên toàn bộ entries, không phải quantile của mean pixel. Std dùng ddof=0 trong thống kê mô tả.")
    r.paragraph("Duplicate group đếm nhóm phổ trùng chính xác; membership đếm mọi pixel trong nhóm, excess đếm bản sao ngoài đại diện đầu. "
                "Zero đơn lẻ không làm pixel invalid. Outlier/duplicate hợp lệ chỉ được flag và giữ trong cube. "
                "CSV theo band ghi original ID, raw range, quantile, zero/nonfinite/>1, signed residual và RMSE; CSV theo pixel giữ ID/row/col/split để truy vết.")
    r.link("tables/band_statistics.csv", "Bảng đủ 198 band")
    r.link("tables/pixel_statistics.csv", "Bảng đủ 10.000 pixel")
    r.figure("scene_overview.png", f"Composite band-index tùy ý RGB retained indices 0-based {s['composite']['retained_band_indices_zero_based_RGB'].tolist()}, "
             f"original IDs {s['composite']['original_band_ids_RGB'].tolist()}. Stretch 2–98% fit train, clipping chỉ trên ảnh hiển thị. "
             "Không gọi true-color/NIR khi chưa có wavelength metadata. Hai panel còn lại là intensity và norm thang nominal.")
    r.figure("qc_maps.png", "QC trên toàn cảnh, pixel flag vẫn giữ. Mọi bản đồ dùng ID/row/col canonical; màu trắng/đen ở mask là trạng thái, không phải nhãn vật liệu.")
    r.heading("3. Cấu trúc phổ, brightness và noise proxies")
    proxy = s["proxies"]
    bright = proxy["brightness_valid"]
    r.paragraph(f"Brightness=mean trên 198 band có mean {bright['mean']:.6g}, median {bright['quantiles']['0.5']:.6g}, "
                f"p99 {bright['quantiles']['0.99']:.6g}. RMS sai phân phổ chỉ dùng {proxy['adjacent_original_gap1_pair_count']} "
                f"cặp original gap=1, bỏ {proxy['excluded_gap_pair_count']} cặp bắc qua band gaps. "
                f"Mean RMS proxy theo pixel là {proxy['adjacent_band_difference_rms_valid']['mean']:.6g}; "
                f"mean spatial-neighbor RMSE proxy là {proxy['spatial_neighbor_rmse_valid']['mean']:.6g}. "
                "Sai phân phổ chứa độ dốc/hấp thụ phổ và variability; sai phân không gian chứa ranh giới vật liệu, texture và chiếu sáng. "
                "Không có phép ước lượng noise độc lập hoặc metadata radiometric để gọi các proxy này là sensor SNR/SNR vật lý.")
    r.table(pd.DataFrame([{"direction": d["direction"], "pair_count": d["n"], "mean_rmse_proxy": d["mean"],
                          "median_rmse_proxy": d["quantiles"]["0.5"], "max_rmse_proxy": d["max"]}
                         for d in proxy["spatial_pairs_by_direction"]]))
    r.figure("spectral_statistics.png", "Toàn cảnh hợp lệ, chỉ mô tả: mean/std/median/5–95%, M reference, histogram brightness theo split và adjacent-band proxy. Đường phổ ngắt tại original band gaps.")
    r.figure("band_correlation.png", "Pearson correlation trên toàn cảnh hợp lệ. Trục original band 1–224; dải trắng là band đã bỏ hoặc correlation không xác định do band hằng. Không fit preprocessing bằng heatmap này.")
    r.link("tables/adjacent_band_difference.csv", "Sai phân chỉ trên cặp original band kề nhau")
    r.heading("4. Reference abundance và coverage calibration")
    r.figure("reference_abundance.png", "Bốn A_ref theo đúng cood local, colorbar cố định 0–1. Đây là reference abundance, không chứng nhận ground truth đo thực địa.")
    r.figure("dominant_and_split.png", "Argmax A_ref là dominant reference map; không gọi classification ground truth. Ties chọn hàng đầu theo thứ tự local. Spatial split giữ nguyên sau EDA.")
    r.figure("purity_entropy_proxies.png", "Purity=max(A_ref); entropy=−sum(a log a)/log(4), quy ước 0log0=0. Các bản đồ proxy không phải physical SNR.")
    r.table(overall[["material", "mean", "median", "std", "mass_fraction_reference", "dominant_count", "pure_ge_0.90", "pure_ge_0.95", "pure_ge_0.99"]],
            "Toàn cảnh hợp lệ, mô tả. Mass fraction ở đây là tỷ trọng tổng abundance reference (sum A_e / sum A); không đo khối lượng vật lý.")
    r.table(abundance[abundance.split != "all_valid_descriptive"][["split", "material", "n", "mean", "dominant_count", "pure_ge_0.90", "pure_ge_0.95", "pure_ge_0.99"]],
            "Coverage từng material × split. Không đổi vùng split dựa trên A_ref hoặc kết quả coverage.")
    r.table(mixtures[mixtures.pure_threshold == .95],
            "Policy khai báo: pure khi max A_ref≥threshold; mixed khi thấp hơn. Trong mixed, interior khi mọi A_ref>0.01; boundary/edge khi min A_ref≤0.01. "
            "Hai lớp mixed rời nhau và cộng lại đúng mixed_count; boundary bao gồm cả gần mặt 2D và cạnh 1D, không chỉ cạnh hình học chính xác. CSV có cả 0.90/0.95/0.99.")
    train_mix = mixtures[(mixtures.split == "train") & (mixtures.pure_threshold == .95)].iloc[0]
    scarce = train_ab.sort_values("pure_ge_0.95").iloc[0]
    r.paragraph(f"Train có {int(train_mix.mixed_count)} mixed pixel tại threshold 0.95, trong đó "
                f"{int(train_mix['mixed_interior_all_a_gt_0.01'])} interior và {int(train_mix['mixed_boundary_min_a_le_0.01'])} boundary; "
                f"pure count={int(train_mix.pure_count)}. Material ít pure train nhất theo 0.95 là {scarce.material}: {int(scarce['pure_ge_0.95'])} pixel. "
                "Đây là căn cứ mô tả khả năng phủ calibration, không phải lý do chọn lại test. Với hard vertex anchor, pure samples có residual parameter gradient bằng 0; "
                "cần mixtures để hiệu chuẩn decoder, đồng thời báo thiếu phủ vùng simplex thay vì suy rộng tự động.")
    r.figure("high_purity_spectra.png", "Pixel được chọn bằng A_ref≥0.95; line đen là mean, dải ±1 std, so với M cùng vật liệu. Selection bằng reference chỉ phục vụ mô tả; n nhỏ không cho chứng nhận endmember variability.")
    r.table(high_purity[(high_purity.split == "all_valid_descriptive") & (high_purity.threshold == .95)],
            "Độ lệch mean phổ high-purity với M; SAD trên phổ chưa center, đơn vị degree. Empty selection được ghi undefined, không thay bằng 0.")
    r.link("tables/high_purity_reference_vs_M.csv", "Coverage và sai khác high-purity đủ ba threshold, mọi split")
    r.heading("5. Hình học endmember và khả năng nhận dạng")
    r.table(pairs)
    r.table(pd.DataFrame([{"matrix": k, "rank": geom[k]["rank"], "singular_values": np.array2string(geom[k]["singular_values"], precision=8),
                          "condition_number": geom[k]["condition_number"], "sigma_min": geom[k]["sigma_min"],
                          "status": geom[k]["condition_status"]} for k in ("M", "MQ")]))
    closest = pairs.sort_values("sad_degrees").iloc[0]
    r.paragraph(f"Cặp góc nhỏ nhất là {closest.material_1}/{closest.material_2}: SAD={closest.sad_degrees:.6g}° "
                f"({closest.sad_radians:.6g} rad), Euclidean distance={closest.euclidean_distance:.6g}. "
                f"sigma_min(MQ)={geom['MQ']['sigma_min']:.8g}; Q là Helmert basis trực chuẩn cho 1ᵀa=0, lưu trong NPZ. "
                "MQ đo sensitivity trên tangent simplex đúng với ràng buộc tổng abundance; condition M và MQ được báo riêng. "
                "SAD nhỏ cho phổ gần cùng hướng, nhưng brightness/độ dài phổ vẫn ảnh hưởng inverse nên phải đọc cùng Euclidean/singular values. "
                "Rank deficiency khiến condition number undefined (JSON null + status). Không áp dụng SAD cho spectra đã PCA-center. "
                "sigma_min(MQ)>0 hỗ trợ tính duy nhất baseline tuyến tính trong precision hiện tại; chưa chứng nhận điều kiện nonlinear sigma>kappa của QSiMix.")
    r.heading("6. PCA chỉ fit train")
    r.paragraph(f"Mean và covariance fit trên {pc['fit_count']} valid train pixels bằng numpy.linalg.eigh; covariance chia n_train−1. "
                f"PC1 giải thích {pc['explained_variance_ratio'][0]*100:.6g}% variance train; ba PC đầu cộng "
                f"{pc['cumulative_explained_variance'][2]*100:.6g}%. Số PC đạt 95/99/99.9% lần lượt là "
                f"{pc['dimensions_for_cumulative_variance']['0.95']}/{pc['dimensions_for_cumulative_variance']['0.99']}/{pc['dimensions_for_cumulative_variance']['0.999']}. "
                "Các split và M cùng transform (X−mean_train), không refit. PCA dimension phản ánh variance dữ liệu trong scale này, "
                "không chứng minh có bấy nhiêu endmember và không phải HySime/MNF. Không dùng PCA reconstruction để thay clean Y.")
    r.figure("pca_train_only.png", "Cumulative variance train và ba panel PC1/2, PC1/3, PC2/3. Display sample seeded trên toàn cảnh, màu theo dominant A_ref chỉ để mô tả; X là M chiếu bằng cùng train mean/components.")
    r.link("tables/pca_explained_variance.csv", "Explained variance đầy đủ 198 components")
    r.heading("7. LMM reference và FCLS thực")
    r.paragraph("R_ref=Y−M A_ref; R_fcls=Y−M A_fcls. Global spectral RMSE=sqrt(mean R²) trên band×pixel trong subset; "
                "per-pixel RMSE lấy mean trên band; per-band RMSE lấy mean trên pixel. SAM dùng uncentered spectra, radian; "
                "zero/nonfinite norm bị mask với số lượng defined được ghi. Global SRE=10log10(sum Y²/sum R²), không lấy mean SRE pixel; "
                "zero signal/error energy là undefined/null kèm status. aRMSE=sqrt(mean((A_fcls−A_ref)²)) chỉ là sai khác với reference.")
    r.table(metrics[["split", "model", "n_pixels", "rmse", "sre_db", "sam_mean_radians", "sam_defined_count", "armse_vs_reference"]])
    reduction = (ref.rmse - fcls.rmse) / ref.rmse * 100 if ref.rmse else 0
    r.paragraph(f"Toàn cảnh, FCLS giảm RMSE phổ {reduction:.4f}% từ {ref.rmse:.8f} xuống {fcls.rmse:.8f}; "
                f"SRE đổi từ {ref.sre_db:.6g} dB sang {fcls.sre_db:.6g} dB. "
                f"Mean SAM đổi từ {ref.sam_mean_radians:.6g} sang {fcls.sam_mean_radians:.6g} rad. "
                "FCLS tối ưu reconstruction trên simplex với M cố định nên cải thiện loss so với A_ref feasible là kỳ vọng toán học; "
                "không chứng minh abundance đo thực địa tốt hơn. Sai khác còn lại có thể do cách tạo reference, M, scale, label correspondence hoặc variability.")
    r.paragraph(f"Solver float64 duyệt đủ {solver['face_count']} nonempty supports (15 với E=4). Mỗi mặt giải equality-constrained LS bằng "
                "a=a0+Qz và lstsq(M_face Q, Y−M_face a0); so sánh objective trực tiếp trên các nghiệm feasible. "
                "Chỉ repair âm cỡ roundoff≤1e−10 rồi bảo toàn ASC, không dùng clip+normalize nghiệm LS unconstrained. "
                f"Max ASC error={solver['asc_max_abs']:.6g}, ANC violation={solver['anc_max_violation']:.6g}, "
                f"max KKT={solver['kkt_max']:.6g}, status={solver['status']}. "
                "KKT dùng objective ||Ma−y||², gradient=2Mᵀ(Ma−y), ν=−mean gradient trên active a>1e−9, μ=g+ν; "
                "báo active stationarity, dual nonnegativity, complementarity và primal feasibility; tolerance tổng 1e−7. "
                "Exact ở đây là xét hết mặt của bài toán convex bằng số float64, không phải số học ký hiệu chính xác.")
    r.paragraph(f"Đối chiếu SLSQP seed={solver['oracle_seed']} trên {solver['oracle_n']} pixel, khởi tạo đều [0.25]*4, "
                f"ftol=1e−12, maxiter=1000: success={solver['oracle_success_count']}/{solver['oracle_n']}; "
                f"max |objective_face−objective_SLSQP|={solver['oracle_max_abs_objective_difference']:.6g}, "
                f"max |A_face−A_SLSQP|={solver['oracle_max_abs_abundance_difference']:.6g}; status={solver['oracle_status']}. "
                "Check phải thỏa objective_face−objective_SLSQP≤1e−8 và max abundance difference≤1e−5 với M local; "
                "abundance của nghiệm rank-deficient nói chung có thể không duy nhất, nên test synthetic suy biến cần so objective/reconstruction.")
    r.link("tables/fcls_slsqp_comparison.csv", "Mọi sample SLSQP, status/message và sai khác objective/abundance")
    r.table(abundance[["split", "material", "fcls_armse_vs_reference", "fcls_bias_vs_reference", "fcls_mae_vs_reference"]],
            "Signed bias=A_fcls−A_ref. Mỗi split có đủ bốn material; không gọi reference error là lỗi so với measured truth.")
    worst = overall.sort_values("fcls_armse_vs_reference", ascending=False).iloc[0]
    r.paragraph(f"Material sai khác reference lớn nhất theo aRMSE toàn cảnh là {worst.material}: "
                f"aRMSE={worst.fcls_armse_vs_reference:.6g}, bias={worst.fcls_bias_vs_reference:.6g}. "
                "Bias cùng bản đồ mismatch giúp nhận ra trao đổi tỷ lệ giữa vật liệu và shift theo không gian; không tự đổi hàng A/cột M để làm residual đẹp hơn.")
    r.figure("fcls_abundance.png", "A_fcls được suy từ Y,M; colorbar 0–1, giữ thứ tự local. Không đưa A_ref vào solver.")
    r.figure("residual_maps.png", "Residual RMSE reference/FCLS dùng cùng color limit; các panel còn lại là mức giảm RMSE, aRMSE-reference và SAM radian trên toàn cảnh hợp lệ.")
    r.figure("fcls_reference_mismatch.png", "Signed abundance mismatch A_fcls−A_ref với color limit đối xứng chung cho bốn material. Đây là sai khác reference, không phải sai số ground truth đo thực địa.")
    r.figure("residual_spectra.png", "Residual theo band và pixel, bias có dấu Y−reconstruction và phổ ba pixel FCLS RMSE cao nhất. Đường phổ không nối qua band gaps; outlier được giữ.")
    r.heading("8. Outlier được giữ và residual confounding")
    r.table(pd.DataFrame([{"quantity": k, **v} for k, v in s["robust_thresholds_train"].items()]),
            "Policy cố định: flag high nếu value>median_train+6×1.4826×MAD_train. Khi MAD=0, threshold undefined và không flag bằng quy tắc này; status ghi rõ. Không auto-delete.")
    r.table(flags)
    show_columns = ["ranking", "rank", "pixel_id", "row", "col", "split_label", "brightness_mean_band", "rmse_reference", "rmse_fcls", "dominant_reference_material", "retained_in_clean"]
    r.table(outliers[outliers["rank"] <= 5][show_columns],
            "Top 5 mỗi ranking (CSV lưu top 20; cùng pixel có thể xuất hiện ở nhiều ranking). Pixel ID,row,col 0-based; split label 0=train,1=val,2=test,3=buffer,4=invalid.")
    r.link("tables/top_outliers_retained.csv", "Top 20 brightness/reference residual/FCLS residual — giữ toàn bộ")
    selected_corr = correlations[(correlations.split == "all_valid_descriptive") &
                                 correlations.feature.isin(["brightness", "purity_A_ref", "entropy_A_ref"])]
    r.table(selected_corr)
    r.table(strata[strata.split == "all_valid_descriptive"],
            "Correlation residual–brightness trong từng dominant-reference stratum. Stratification vẫn dùng A_ref, chưa loại confounding hoàn toàn.")
    for model in ("reference_rmse", "fcls_rmse"):
        subset = selected_corr[selected_corr.residual == model].set_index("feature")
        r.paragraph(f"{model}: Pearson với brightness={subset.loc['brightness', 'pearson']:.6g}, "
                    f"purity={subset.loc['purity_A_ref', 'pearson']:.6g}, entropy={subset.loc['entropy_A_ref', 'pearson']:.6g}; "
                    f"Spearman brightness={subset.loc['brightness', 'spearman']:.6g}. "
                    "Các hệ số này mô tả đồng biến trong scene đã quan sát; không có kiểm định causal hoặc giả thiết pixel độc lập.")
    r.paragraph("Residual tổng hợp sensor noise, lượng tử hóa raw, nominal scale, endmember variability, shade/brightness, "
                "reference-label mismatch và khả năng nonlinear mixing. Correlation với entropy không cô lập được phi tuyến: composition, "
                "độ sáng và cấu trúc không gian có thể cùng thay đổi. Spatial autocorrelation khiến p-value iid dễ gây hiểu nhầm; báo cáo chỉ đưa hệ số, "
                "sample counts và bảng theo split/stratum. Pixel outlier là ứng viên xem lại phổ/ảnh, chưa có bằng chứng để xóa hoặc gán lỗi vật lý.")
    r.figure("residual_confounding.png", "Scatter residual với brightness/purity/entropy, mẫu hiển thị seeded; màu dominant reference. Các hệ số trong bảng dùng toàn bộ valid pixels của subset, không chỉ display sample.")
    r.heading("9. Kết luận định lượng cho QSiMix-Residual")
    r.paragraph(f"Calibration: train có {int(train_mix.mixed_count)} mixtures tại pure threshold 0.95 và coverage khác nhau giữa tree/water/dirt/road. "
                "Bảng pure theo material/split phải đi cùng mọi thử nghiệm mixture calibration; pure endpoints một mình không hiệu chuẩn nonlinear residual trong hard-anchor model. "
                "Cần giữ baseline FCLS và đối chứng classical anchored residual trong cùng scale/protocol, báo đồng thời reconstruction và abundance-reference mismatch.")
    r.paragraph(f"Khả năng nhận dạng: cặp {closest.material_1}/{closest.material_2} gần hướng phổ nhất ({closest.sad_degrees:.4f}°); "
                f"sigma_min(MQ)={geom['MQ']['sigma_min']:.6g}. Đây là mốc sensitivity tuyến tính để đọc mức sai khác abundance, "
                "không phải nonlinear inverse certificate. So sánh high-purity với M cho ứng viên variability/anchor mismatch; selection dùng generated reference "
                "không chứng minh vật liệu thuần vật lý hoặc nguyên nhân chênh lệch. Không sửa M trong EDA.")
    r.paragraph(f"Miền mô hình: {inv['gt1_pixels']} pixel có Y>1 (max={inv['clean_Y']['max']:.6g}) gây xung đột với pilot [0,1]. "
                "Trước training cần adapter hoặc nới miền có khai báo. Nếu chọn một scalar c từ train, áp dụng đồng bộ Y/c, M/c, residual/c và lambda/c; "
                "không clip test để ép miền và không tự đổi lambda trong task này. Brightness/proxy noise hiện tại không hiệu chuẩn physical SNR. "
                "Reference labels và M có hạn chế nguồn; kết quả trên split exploratory này chỉ hỗ trợ phát triển pipeline, cần đánh giá confirmatory độc lập cho claim nghiên cứu.")
    r.heading("10. Tái lập và deliverable contract")
    r.paragraph("Chạy từ repo root: .venv-jasper/Scripts/python.exe scripts/eda_jasper.py --root . "
                "Input bắt buộc: data/processed/jasper_ridge/jasper_clean.npz và manifest.json đúng §4. "
                "Matplotlib Agg; không notebook, sklearn hoặc mạng để render. HTML chỉ dùng CSS inline và PNG/CSV local; chuyển cả thư mục reports/jasper_ridge để đọc hình offline. "
                "Hai link source audit/QA phụ thuộc artifact agents khác; manifest link cần giữ cấu trúc repo.")
    r.paragraph(f"Seed={s['seed']}; versions={s['versions']}. Input NPZ SHA256={s['input_npz_sha256']}; "
                f"manifest SHA256={s['input_manifest_sha256']}. Sign PCA được cố định theo loading lớn nhất; eigenspace suy biến có thể quay theo thư viện số. "
                "PCA và inverse tính float64; kết quả arrays và split tái lập trong cùng environment. Raw chỉ đọc, code EDA không ghi preprocessing hoặc source.")
    r.paragraph("eda_summary.json: inventory là QC toàn cảnh; geometry chứa pairwise SAD rad/degree, singular values/rank/condition M,MQ; "
                "pca là fit valid train; reconstruction_by_split ghi spectral RMSE/SRE/SAM, abundance_by_material_split ghi mean/coverage/aRMSE/bias; "
                "fcls chứa 15 faces, KKT và SLSQP validation; robust_thresholds_train là fitted policy; residual_confounding_correlations là descriptive. "
                "JSON chuẩn allow_nan=False: undefined thành null kèm status; CSV undefined là ô trống. NPZ dùng NaN cho pixel invalid hoặc angle undefined, không object arrays.")
    r.paragraph("eda_arrays.npz: A_fcls và A_fcls_minus_A_ref [4,N]; residual_reference/residual_fcls [B,N]; "
                "rmse_*_per_pixel, sam_*_radians_per_pixel, armse_fcls_vs_reference_per_pixel và diagnostics FCLS [N]. "
                "pca_mean_train [B], pca_components_train [K,B], pca_scores_all_pixels [K,N], pca_endmember_scores [K,4], K=198; "
                "scores=components@(Y−mean[:,None]). pca_fit_pixel_ids lưu đúng train IDs, pca_explained_variance*_train là train only. "
                "simplex_tangent_Q [4,3]; maps/proxies/flags [N], band_correlation_all_valid_descriptive [B,B]. "
                "Mọi per-pixel array cùng hệ ID gốc; row/col/split_labels/band_ids_original/material_names lưu kèm. "
                "fcls_support_bitmask dùng bit 0=tree,1=water,2=dirt,3=road; fcls_objective là sum squared spectral residual; "
                "fcls_*stationarity/dual_violation/complementarity là KKT theo convention §7. array_contract trong JSON liệt kê dtype/shape từng key.")
    for path, label in (("eda_summary.json", "JSON số liệu và array schema"), ("eda_arrays.npz", "NPZ PCA/FCLS/residual/maps")):
        r.link(path, label)
    for path in sorted((target / "tables").glob("*.csv")):
        r.link("tables/" + path.name, path.name)
    r.write(target)

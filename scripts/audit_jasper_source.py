"""Read-only Jasper Ridge MAT/FIG source audit; prints JSON to stdout only."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import scipy
from scipy.io import loadmat, whosmat


ROOT = Path(__file__).resolve().parents[1]


def stats(value):
    x = np.asarray(value)
    finite = x[np.isfinite(x)]
    if np.iscomplexobj(x):
        return {"shape": list(x.shape), "dtype": str(x.dtype),
                "real": stats(x.real), "imag": stats(x.imag)}
    return {
        "shape": list(x.shape), "dtype": str(x.dtype), "size": int(x.size),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "nan": int(np.isnan(x).sum()), "inf": int(np.isinf(x).sum()),
        "zeros": int((x == 0).sum()),
    }


def walk(value, path=""):
    if hasattr(value, "_fieldnames"):
        for key in value._fieldnames:
            yield from walk(getattr(value, key), path + "/" + key)
    elif isinstance(value, dict):
        for key, child in value.items():
            if not key.startswith("__"):
                yield from walk(child, path + "/" + key)
    elif isinstance(value, (tuple, list)):
        for i, child in enumerate(value):
            yield from walk(child, path + "/" + str(i))
    elif isinstance(value, np.ndarray) and value.dtype == object:
        for i, child in enumerate(value.flat):
            yield from walk(child, path + "/" + str(i))
    else:
        yield path, value


def inspect_file(path):
    # Preserve numeric array shapes and dtypes; simplify_cells squeezes scalars.
    data = loadmat(path, struct_as_record=False)
    entries = {}
    for key, value in walk(data):
        x = np.asarray(value)
        if np.issubdtype(x.dtype, np.number):
            entry = stats(x)
            if x.size <= 20 and not np.iscomplexobj(x):
                entry["values"] = x.tolist()
            entries[key] = entry
        else:
            entries[key] = x.tolist()
    return {
        "file": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "header": str(data.get("__header__")),
        "keys": whosmat(path), "entries": entries,
    }


def figure_objects(path, object_type, field):
    root = loadmat(path, simplify_cells=True)["hgS_070000"]
    found = []
    for i, ax in enumerate(root["children"]):
        children = ax.get("children", [])
        if isinstance(children, dict):
            children = [children]
        labels = [ch["properties"]["String"] for ch in children
                  if ch.get("type") == "text"
                  and isinstance(ch.get("properties", {}).get("String"), str)
                  and ch["properties"]["String"]]
        for j, child in enumerate(children):
            if child.get("type") == object_type:
                found.append({"path": f"hgS_070000.children[{i}].children[{j}]",
                              "labels": labels, "data": child["properties"][field],
                              "object_properties": child["properties"],
                              "axes_properties": ax["properties"]})
    return found


def matching(reference, candidates):
    # Rows: GT class; columns: FIG axes order. Exhaust all 4! permutations.
    errors = np.array([[np.max(np.abs(a - b)) for b in candidates] for a in reference])
    ranked = sorted((max(float(errors[i, p[i]]) for i in range(4)), p)
                    for p in itertools.permutations(range(4)))
    return {"max_abs_error_matrix": errors.tolist(),
            "best_permutation_zero_based": list(ranked[0][1]),
            "best_max_abs_error": ranked[0][0],
            "runner_up_max_abs_error": ranked[1][0],
            "exact_permutations": [list(p) for error, p in ranked if error == 0]}


def comparisons():
    gt = loadmat(ROOT / "data/Jasper_GT.mat")
    source = loadmat(ROOT / "data/jasperRidge2_R198.mat")
    a, m, y = gt["A"], gt["M"], source["Y"]
    abundance = figure_objects(ROOT / "data/end4_Abundance.fig", "image", "CData")
    materials = figure_objects(ROOT / "data/end4_Materials.fig", "graph2d.lineseries", "YData")
    sums = a.sum(axis=0)
    bands = source["SlectBands"].ravel().astype(int)
    divisor = float(source["maxValue"].item())
    result = {
        "presence": {"Jasper_GT.mat": {k: k in gt for k in ("M", "A", "Y")},
                     "jasperRidge2_R198.mat": {k: k in source for k in ("M", "A", "Y")}},
        "labels": loadmat(ROOT / "data/Jasper_GT.mat", simplify_cells=True)["cood"].tolist(),
        "abundance_sum": {**stats(sums), "max_abs_error_from_one": float(np.max(np.abs(sums - 1))),
                          "not_exact_one": int(np.count_nonzero(sums != 1)),
                          "beyond_1e_12": int(np.count_nonzero(np.abs(sums - 1) > 1e-12))},
        "A_negative": int((a < 0).sum()), "A_above_one": int((a > 1).sum()),
        "A_zero_columns": int(np.all(a == 0, axis=0).sum()),
        "Y_zero_columns": int(np.all(y == 0, axis=0).sum()),
        "Y_zero_rows": int(np.all(y == 0, axis=1).sum()),
        "Y_above_maxValue": int((y > divisor).sum()),
        "Y_div_metadata_maxValue": stats(y.astype(np.float64) / divisor),
        "selected_bands": bands.tolist(),
        "selected_unique": int(np.unique(bands).size),
        "selected_strictly_increasing": bool(np.all(np.diff(bands) > 0)),
        "excluded_bands_1_based": sorted(set(range(1, int(source["nBand"].item()) + 1)) - set(bands.tolist())),
        "M_vs_fig": matching(m.T, [obj["data"] for obj in materials]),
        "A_vs_fig": {order: matching(a, [obj["data"].ravel(order=order) for obj in abundance])
                     for order in ("F", "C")},
        "fig_materials": [{"path": obj["path"], "labels": obj["labels"],
                           "YData": stats(obj["data"]),
                           "XData_is_1_to_198": bool(np.array_equal(obj["object_properties"]["XData"], np.arange(1, 199)))}
                          for obj in materials],
        "fig_abundance": [{"path": obj["path"], "labels": obj["labels"],
                           "CData": stats(obj["data"]),
                           "XData": obj["object_properties"]["XData"].tolist(),
                           "YData": obj["object_properties"]["YData"].tolist(),
                           "YDir": obj["axes_properties"].get("YDir", "<not stored>"),
                           "CLim": obj["axes_properties"]["CLim"].tolist()}
                          for obj in abundance],
    }
    # No regression against M/A, no scale optimization, no pipeline or raw writes.
    return result


def git_evidence():
    commands = [["remote", "-v"], ["rev-parse", "HEAD"],
                ["log", "--all", "--format=%H %s", "--", "data"],
                ["ls-tree", "-r", "HEAD", "data"], ["diff", "--name-only", "HEAD", "--", "data"]]
    evidence = {}
    for command in commands:
        proc = subprocess.run(["git", *command], cwd=ROOT, capture_output=True, text=True, check=False)
        evidence["git " + " ".join(command)] = {"returncode": proc.returncode,
                                                   "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true", help="Include all nested fields")
    args = parser.parse_args()
    paths = sorted(p for p in (ROOT / "data").rglob("*")
                   if p.suffix.lower() in {".mat", ".fig"})
    result = {"python": platform.python_version(), "numpy": np.__version__,
              "scipy": scipy.__version__, "files": [inspect_file(p) for p in paths],
              "comparisons": comparisons(), "git": git_evidence()}
    if not args.inspect:
        for item in result["files"]:
            if item["file"].endswith(".fig"):
                item["entries"] = {k: v for k, v in item["entries"].items()
                                   if k.lower().endswith(("/type", "/string", "/tag", "/displayname", "/cdata", "/xdata", "/ydata", "/clim", "/ydir", "/name"))}
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

"""Independent numerical acceptance checks for plan.md §§4–7.

Run only this file: the legacy src package imports a missing src.config.
Oracles below read source MAT/FIG directly and never import the source audit.
Missing implementation/artifacts fail explicitly, rather than count as passes.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.io import loadmat
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RAW_HASHES = {
    "jasperRidge2_R198.mat": "0e4118a6452f6044978a8ca3762fb0f791115467904936d463c4e111e56e682e",
    "Jasper_GT.mat": "92f5697b43705802b904fd13ba99b6ce65a3d203682864abc3fbec922beec374",
    "end4_Abundance.fig": "1d0ce4237d03ac97cd7f64debde8e79d2a526f050357a8bdbc0efdee8be2b857",
    "end4_Materials.fig": "7870d2be51e88da548665ff5093a0216f2c894d5630464a0219c395b829a13ad",
}
MATERIALS = ["tree", "water", "dirt", "road"]
# Explicit independent set subtraction; never derive expected bands from code under test.
EXCLUDED_BANDS = {1, 2, 3, *range(108, 113), *range(154, 167), *range(220, 225)}
EXPECTED_BANDS = np.array([i for i in range(1, 225) if i not in EXCLUDED_BANDS], dtype=np.int64)
PROCESSED = ROOT / "data/processed/jasper_ridge"
REPORTS = ROOT / "reports/jasper_ridge"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json(path):
    def reject_constant(value):
        raise AssertionError(f"Non-standard JSON numeric constant: {value}")

    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)

    def check(node):
        if isinstance(node, float):
            assert np.isfinite(node), "Overflow/nonfinite number in JSON"
        elif isinstance(node, dict):
            for child in node.values():
                check(child)
        elif isinstance(node, list):
            for child in node:
                check(child)

    check(value)
    return value


def read_npz(path):
    assert path.is_file(), f"Required artifact absent: {path}"
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


@pytest.fixture(scope="session")
def raw():
    return loadmat(ROOT / "data/jasperRidge2_R198.mat"), loadmat(ROOT / "data/Jasper_GT.mat")


@pytest.fixture(scope="session")
def clean():
    return read_npz(PROCESSED / "jasper_clean.npz")


@pytest.fixture(scope="session")
def eda_arrays():
    return read_npz(REPORTS / "eda_arrays.npz")


@pytest.fixture(scope="session")
def preprocess():
    return importlib.import_module("jasper_preprocess")


@pytest.fixture(scope="session")
def eda():
    return importlib.import_module("jasper_eda")


@pytest.mark.parametrize("name,expected", RAW_HASHES.items())
def test_raw_files_keep_pinned_preprocessing_hashes(name, expected):
    assert sha256(ROOT / "data" / name) == expected


def test_exact_source_selected_bands(raw):
    spectra, _ = raw
    assert_array_equal(spectra["SlectBands"].ravel(), EXPECTED_BANDS)
    assert spectra["Y"].shape == (198, 10000)
    assert spectra["Y"].dtype == np.uint16
    assert spectra["nBand"].item() == 224  # historical metadata, not current band count


@pytest.mark.parametrize("k,name", list(enumerate(MATERIALS)))
def test_fig_reference_alignment_and_names(raw, k, name):
    """This proves A/M versus FIG; it does NOT establish Y versus A registration."""
    _, reference = raw
    abundance = loadmat(ROOT / "data/end4_Abundance.fig", simplify_cells=True)["hgS_070000"]
    materials = loadmat(ROOT / "data/end4_Materials.fig", simplify_cells=True)["hgS_070000"]
    abundance_children = abundance["children"][k]["children"]
    material_children = materials["children"][k]["children"]
    label = f"{k + 1}-{name}"
    assert abundance_children[1]["properties"]["String"] == label
    assert material_children[1]["properties"]["String"] == label
    source_label = np.asarray(reference["cood"][k, 0]).item()
    assert source_label == label
    image = abundance_children[0]["properties"]["CData"]
    assert_array_equal(image, reference["A"][k].reshape((100, 100), order="F"))
    assert not np.array_equal(image, reference["A"][k].reshape((100, 100), order="C"))
    assert_array_equal(material_children[0]["properties"]["YData"], reference["M"][:, k])


def test_clean_schema_exact_values_no_double_drop_or_scale(clean, raw):
    spectra, reference = raw
    required = {"Y", "M", "A_ref", "band_ids_original", "pixel_ids", "row", "col",
                "valid_pixel_mask", "train_idx", "val_idx", "test_idx", "buffer_idx",
                "split_labels", "material_names", "height", "width"}
    assert required <= clean.keys()
    for key, shape in [("Y", (198, 10000)), ("M", (198, 4)), ("A_ref", (4, 10000))]:
        assert clean[key].shape == shape
        assert clean[key].dtype == np.float64
    assert_array_equal(clean["Y"], spectra["Y"].astype(np.float64) / 5000.0)
    assert_array_equal(clean["M"], reference["M"])
    assert_array_equal(clean["A_ref"], reference["A"])
    assert_allclose(clean["Y"] * 5000.0, spectra["Y"], atol=1e-12, rtol=0)
    assert clean["Y"].max() == 5437 / 5000
    assert np.count_nonzero(clean["Y"] > 1) == 18
    assert np.count_nonzero(clean["Y"] == 0) == 418
    assert clean["material_names"].dtype.kind == "U"
    assert clean["material_names"].tolist() == MATERIALS
    assert_array_equal(clean["band_ids_original"], EXPECTED_BANDS)
    for key in ["band_ids_original", "pixel_ids", "row", "col", "train_idx", "val_idx", "test_idx", "buffer_idx"]:
        assert clean[key].dtype == np.int64, key
    assert clean["valid_pixel_mask"].dtype == np.bool_
    assert clean["valid_pixel_mask"].shape == (10000,)
    assert clean["valid_pixel_mask"].all()  # lone zeros are valid
    for key in ["height", "width"]:
        assert clean[key].shape == ()
        assert clean[key].dtype.kind in "iu"
        assert clean[key].item() == 100


def test_pixel_ids_and_spatial_split_independent_oracle(clean):
    ids = np.arange(10000, dtype=np.int64)
    # Independent integer-coordinate construction, without a reshape roundtrip oracle.
    rows = ids % 100
    cols = ids // 100
    assert_array_equal(clean["pixel_ids"], ids)
    assert_array_equal(clean["row"], rows)
    assert_array_equal(clean["col"], cols)
    expected = np.full(10000, 3)
    expected[cols <= 57] = 0
    expected[(cols >= 60) & (cols <= 77)] = 1
    expected[cols >= 80] = 2
    expected[~clean["valid_pixel_mask"]] = 4
    assert_array_equal(clean["split_labels"], expected)
    groups = [clean[k] for k in ("train_idx", "val_idx", "test_idx", "buffer_idx")]
    for label, group in enumerate(groups):
        assert_array_equal(np.sort(group), ids[expected == label])
        assert len(group) == len(np.unique(group))
    joined = np.concatenate(groups)
    assert len(np.unique(joined)) == len(joined)
    assert_array_equal(np.sort(joined), ids[clean["valid_pixel_mask"]])
    assert [len(group) for group in groups] == [5800, 1800, 2000, 400]
    for first, second in [(groups[0], groups[1]), (groups[1], groups[2]), (groups[0], groups[2])]:
        distance = np.abs(np.unique(cols[first])[:, None] - np.unique(cols[second])[None, :])
        assert distance.min() >= 3  # two entire excluded columns between sets


def test_export_json_strict_and_manifest_references_raw_hashes():
    manifest = strict_json(PROCESSED / "manifest.json")
    encoded = json.dumps(manifest, ensure_ascii=False)
    for name, digest in RAW_HASHES.items():
        assert name in encoded
        assert digest in encoded
    assert sha256(PROCESSED / "jasper_clean.npz") in encoded


def test_eda_deliverables_exist_and_json_is_standard():
    summary = strict_json(REPORTS / "eda_summary.json")
    assert isinstance(summary, dict) and summary
    for name in ["eda_report.md", "eda_report.html", "eda_arrays.npz"]:
        path = REPORTS / name
        assert path.is_file() and path.stat().st_size > 100, name
    assert list((REPORTS / "tables").glob("*.csv")), "No EDA tables"
    assert list((REPORTS / "figures").glob("*.png")), "No EDA figures"


def slsqp_oracle(M, y):
    """Separate constrained optimizer; no reference labels or production helpers."""
    initial = np.full(M.shape[1], 1.0 / M.shape[1])
    result = minimize(
        lambda a: np.dot(M @ a - y, M @ a - y),
        initial,
        jac=lambda a: 2 * M.T @ (M @ a - y),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * M.shape[1],
        constraints={"type": "eq", "fun": lambda a: a.sum() - 1,
                     "jac": lambda a: np.ones_like(a)},
        options={"ftol": 1e-13, "maxiter": 2000},
    )
    assert result.success, result.message
    return result.x, result.fun


@pytest.fixture
def isolated_root(tmp_path):
    """Generated outputs stay in pytest temporary storage, never the live export."""
    (tmp_path / "data").mkdir()
    for name in RAW_HASHES:
        shutil.copy2(ROOT / "data" / name, tmp_path / "data" / name)
    return tmp_path


def test_config_pins_measured_hashes():
    config = strict_json(ROOT / "configs/jasper_preprocess.json")
    assert config["expected_raw_sha256"] == {f"data/{name}": digest for name, digest in RAW_HASHES.items()}


def test_nonsquare_known_pixel_mapping_and_roundtrip(preprocess):
    height, width, bands = 3, 5, 2
    ids, rows, cols = preprocess.spatial_indices(height, width)
    # Literal grid is the orientation oracle; it is not built using the tested helper.
    expected_grid = np.array([[0, 3, 6, 9, 12], [1, 4, 7, 10, 13], [2, 5, 8, 11, 14]])
    actual_grid = np.full((height, width), -1)
    actual_grid[rows, cols] = ids
    assert_array_equal(actual_grid, expected_grid)
    spectra = np.stack([np.arange(15), 100 + np.arange(15)]).astype(np.float64)
    cube = spectra.T.reshape(height, width, bands, order="F")
    assert_array_equal(cube[:, :, 0], expected_grid)
    assert_array_equal(cube[:, :, 1], 100 + expected_grid)
    assert_array_equal(cube[1, 3], [10, 110])
    assert_array_equal(cube[rows, cols], spectra.T)
    assert_array_equal(cube.reshape(height * width, bands, order="F").T, spectra)


def test_invalid_mask_policy_keeps_dark_zero_entry_and_duplicate_spectra(preprocess):
    y = np.array([[0, 0, -1, np.nan, 1, 1e-14, 0, np.inf],
                  [1, 0,  2,      1, 2, 1e-14, 1,     0]], dtype=float)
    before = y.copy()
    masks = preprocess.invalid_reason_masks(y)
    assert_array_equal(masks["nonfinite_spectrum"], [False, False, False, True, False, False, False, True])
    assert_array_equal(masks["negative_spectrum"], [False, False, True, False, False, False, False, False])
    assert_array_equal(masks["all_zero_spectrum"], [False, True, False, False, False, False, False, False])
    assert_array_equal(~np.logical_or.reduce(list(masks.values())), [True, False, False, False, True, True, True, False])
    assert_array_equal(y, before)


def test_preprocess_reproducibility_and_raw_immutability(preprocess, isolated_root, clean):
    config = strict_json(ROOT / "configs/jasper_preprocess.json")
    before = {name: sha256(isolated_root / "data" / name) for name in RAW_HASHES}
    first_manifest = preprocess.prepare_dataset(isolated_root, config)
    output = isolated_root / "data/processed/jasper_ridge/jasper_clean.npz"
    first = read_npz(output)
    second_manifest = preprocess.prepare_dataset(isolated_root, config)
    second = read_npz(output)
    assert first.keys() == second.keys() == clean.keys()
    for key in first:
        assert_array_equal(first[key], second[key], err_msg=key)
        assert_array_equal(first[key], clean[key], err_msg=key)
        assert first[key].dtype == second[key].dtype == clean[key].dtype
    assert {name: sha256(isolated_root / "data" / name) for name in RAW_HASHES} == before == RAW_HASHES
    for manifest in [first_manifest, second_manifest]:
        assert manifest["raw_integrity"]["unchanged"] is True
        assert manifest["splits"]["counts"] == {"train": 5800, "val": 1800, "test": 2000, "buffer": 400, "invalid": 0}
    assert first_manifest["quality"] == second_manifest["quality"]


def test_invalid_pixels_excluded_from_every_split_without_deleting_columns(preprocess, isolated_root, raw, monkeypatch):
    spectra, reference = raw
    modified = dict(spectra, Y=spectra["Y"].copy())
    invalid_ids = np.array([0, 5800, 6000, 8000])  # one each train/buffer/val/test
    modified["Y"][:, invalid_ids] = 0
    monkeypatch.setattr(preprocess, "loadmat", lambda path: modified if Path(path).name == "jasperRidge2_R198.mat" else reference)
    config = strict_json(ROOT / "configs/jasper_preprocess.json")
    manifest = preprocess.prepare_dataset(isolated_root, config)
    result = read_npz(isolated_root / "data/processed/jasper_ridge/jasper_clean.npz")
    assert result["Y"].shape == (198, 10000)
    assert_array_equal(np.flatnonzero(~result["valid_pixel_mask"]), invalid_ids)
    assert_array_equal(result["split_labels"][invalid_ids], [4, 4, 4, 4])
    groups = [result[key] for key in ("train_idx", "val_idx", "test_idx", "buffer_idx")]
    assert_array_equal(np.sort(np.concatenate(groups)), np.setdiff1d(np.arange(10000), invalid_ids))
    assert manifest["masks"]["invalid_count"] == 4
    assert_array_equal(result["M"], reference["M"])
    assert_array_equal(result["A_ref"], reference["A"])


@pytest.mark.parametrize("bad_case", ["missing_Y", "transposed_Y", "float_Y", "bands", "names", "negative_M", "nonfinite_A", "bad_ASC"])
def test_preprocess_fail_fast_malformed_loader_output(preprocess, isolated_root, raw, monkeypatch, bad_case):
    spectra, reference = ({k: v.copy() if isinstance(v, np.ndarray) else v for k, v in part.items()} for part in raw)
    if bad_case == "missing_Y":
        spectra.pop("Y")
    elif bad_case == "transposed_Y":
        spectra["Y"] = spectra["Y"].T
    elif bad_case == "float_Y":
        spectra["Y"] = spectra["Y"].astype(float)
    elif bad_case == "bands":
        spectra["SlectBands"][0, 0] = 5
    elif bad_case == "names":
        reference["cood"] = reference["cood"][[1, 0, 2, 3]]
    elif bad_case == "negative_M":
        reference["M"][0, 0] = -1e-4
    elif bad_case == "nonfinite_A":
        reference["A"][0, 0] = np.nan
    elif bad_case == "bad_ASC":
        reference["A"][:, 0] = .1
    monkeypatch.setattr(preprocess, "loadmat", lambda path: spectra if Path(path).name == "jasperRidge2_R198.mat" else reference)
    with pytest.raises(ValueError):
        preprocess.prepare_dataset(isolated_root, strict_json(ROOT / "configs/jasper_preprocess.json"))
    assert not (isolated_root / "data/processed/jasper_ridge/jasper_clean.npz").exists()


def assert_pca_covariance(model, y, train):
    expected_mean = np.mean(y[:, train], axis=1)
    expected_covariance = np.cov(y[:, train], ddof=1)
    c, values = model["components"], model["explained_variance"]
    assert_allclose(model["mean"], expected_mean, atol=1e-14, rtol=1e-13)
    assert_allclose(c.T @ np.diag(values) @ c, expected_covariance, atol=2e-13, rtol=1e-11)
    assert_allclose(c @ c.T, np.eye(y.shape[0]), atol=1e-12, rtol=0)
    # SVD is independent of the production symmetric eigendecomposition.
    singular = np.linalg.svd(y[:, train] - expected_mean[:, None], compute_uv=False)
    assert_allclose(values, singular ** 2 / (len(train) - 1), atol=2e-13, rtol=1e-10)
    assert_allclose(model["explained_variance_ratio"], values / values.sum(), atol=1e-14)


def test_pca_train_only_independent_covariance_and_heldout_perturbation(eda):
    rng = np.random.default_rng(711)
    y = rng.normal(size=(6, 39)) * np.arange(1, 7)[:, None]
    train = np.array([0, 2, 4, 5, 8, 9, 13, 17, 19, 22, 26, 28, 31, 35, 38])
    original = y.copy()
    fitted = eda.fit_train_pca(y, train)
    assert_pca_covariance(fitted, y, train)
    assert_array_equal(y, original)
    heldout = np.setdiff1d(np.arange(y.shape[1]), train)
    perturbed = y.copy()
    perturbed[:, heldout] += 10000 * np.arange(1, 7)[:, None]
    changed = eda.fit_train_pca(perturbed, train)
    for key in ["mean", "components", "explained_variance", "explained_variance_ratio", "fit_pixel_ids"]:
        assert_array_equal(fitted[key], changed[key], err_msg=key)
    perturbed[:, heldout] = np.nan
    assert_array_equal(eda.fit_train_pca(perturbed, train)["mean"], fitted["mean"])


def test_robust_threshold_zero_mad_is_undefined_and_train_only(eda):
    train = np.array([0, 2, 4, 6])
    constant_train = np.array([2, 900, 2, 1000, 2, 1100, 2, 1200], dtype=float)
    result = eda._robust_threshold(constant_train, train)
    assert result["mad"] == 0
    assert result["threshold"] is None
    assert result["status"] == "undefined_zero_mad"
    assert result["fit_n"] == len(train)
    constant_train[train] = [1, 2, 4, 8]
    result = eda._robust_threshold(constant_train, train)
    median = np.median([1, 2, 4, 8])
    mad = np.median(np.abs(np.array([1, 2, 4, 8]) - median))
    assert result["threshold"] == median + 6 * 1.4826 * mad
    constant_train[[1, 3, 5, 7]] *= 1e8
    assert eda._robust_threshold(constant_train, train) == result


def test_exported_pca_matches_train_covariance_and_transforms(clean, eda_arrays):
    model = {"mean": eda_arrays["pca_mean_train"], "components": eda_arrays["pca_components_train"],
             "explained_variance": eda_arrays["pca_explained_variance_train"],
             "explained_variance_ratio": eda_arrays["pca_explained_variance_ratio_train"]}
    assert_array_equal(eda_arrays["pca_fit_pixel_ids"], clean["train_idx"])
    assert_pca_covariance(model, clean["Y"], clean["train_idx"])
    assert not np.allclose(model["mean"], clean["Y"].mean(axis=1), atol=1e-5)
    assert_allclose(eda_arrays["pca_scores_all_pixels"], model["components"] @ (clean["Y"] - model["mean"][:, None]), atol=2e-12)
    assert_allclose(eda_arrays["pca_endmember_scores"], model["components"] @ (clean["M"] - model["mean"][:, None]), atol=2e-12)


def test_fcls_synthetic_recovery_vertices_faces_interior_and_immutable_inputs(eda):
    rng = np.random.default_rng(202601)
    m = rng.uniform(.05, .9, size=(13, 4))
    # Every nonempty support, including the four exact vertices and full interior.
    abundances = []
    for bitmask in range(1, 16):
        weights = rng.uniform(.2, 1, 4) * np.array([bool(bitmask & (1 << i)) for i in range(4)])
        abundances.append(weights / weights.sum())
    truth = np.array(abundances).T
    y = m @ truth
    m_before, y_before = m.copy(), y.copy()
    result, diagnostic = eda.fcls_exact(y, m)
    assert_allclose(result, truth, atol=2e-12, rtol=0)
    assert_allclose(m @ result, y, atol=2e-12, rtol=0)
    assert diagnostic["face_count"] == 15
    assert diagnostic["kkt_max"] < 1e-10
    assert_array_equal(m, m_before)
    assert_array_equal(y, y_before)
    assert_array_equal(eda.solve_fcls(y, m), result)


def test_fcls_is_not_clip_and_renormalize_and_is_permutation_equivariant(eda):
    m = np.eye(4)
    y = np.array([[.8], [.4], [.2], [-.1]])
    expected = np.array([[2 / 3], [4 / 15], [1 / 15], [0]])
    result = eda.solve_fcls(y, m)
    assert_allclose(result, expected, atol=1e-12, rtol=0)
    clipped = np.maximum(y, 0) / np.maximum(y, 0).sum()
    assert np.linalg.norm(result - clipped) > .05
    permutation = [2, 0, 3, 1]
    assert_allclose(eda.solve_fcls(y, m[:, permutation]), result[permutation], atol=1e-12)


@pytest.mark.parametrize("degenerate", [False, True])
def test_fcls_noisy_synthetic_objective_against_slsqp(eda, degenerate):
    rng = np.random.default_rng(1314)
    m = rng.uniform(0, 1, (11, 4))
    if degenerate:
        m[:, 3] = m[:, 1]  # coefficients need not be unique; objective must be optimal
    y = m @ rng.dirichlet(np.ones(4), 12).T + rng.normal(0, .12, (11, 12))
    fitted = eda.solve_fcls(y, m)
    assert np.min(fitted) >= -1e-12
    assert_allclose(fitted.sum(axis=0), 1, atol=1e-12)
    for pixel in range(y.shape[1]):
        oracle_a, oracle_obj = slsqp_oracle(m, y[:, pixel])
        actual_obj = np.sum((m @ fitted[:, pixel] - y[:, pixel]) ** 2)
        assert abs(actual_obj - oracle_obj) < 2e-10
        if not degenerate:
            assert_allclose(fitted[:, pixel], oracle_a, atol=2e-6, rtol=0)


def test_fcls_real_sample_independent_slsqp_and_export(clean, eda, eda_arrays):
    # Different seed from the production sample, plus boundary and brightest pixels.
    sampled = np.random.default_rng(19337).choice(clean["pixel_ids"], size=40, replace=False)
    sample = np.unique(np.r_[sampled, 0, 5799, 5800, 5999, 6000, 7799, 7800, 7999, 8000, 9999,
                             np.argmax(clean["Y"].mean(axis=0))])
    y, m = clean["Y"][:, sample], clean["M"]
    fitted = eda.solve_fcls(y, m)  # labels are never provided
    assert_allclose(fitted, eda_arrays["A_fcls"][:, sample], atol=1e-11, rtol=0)
    for j, pixel in enumerate(sample):
        oracle_a, oracle_obj = slsqp_oracle(m, y[:, j])
        actual_obj = np.sum((m @ fitted[:, j] - y[:, j]) ** 2)
        assert abs(actual_obj - oracle_obj) < 1e-9, int(pixel)
        assert_allclose(fitted[:, j], oracle_a, atol=3e-6, rtol=0, err_msg=f"pixel={pixel}")


def test_eda_residual_abundance_maps_and_global_kkt_independent(clean, eda_arrays):
    y, m, a = clean["Y"], clean["M"], clean["A_ref"]
    af = eda_arrays["A_fcls"]
    assert af.shape == a.shape and af.dtype == np.float64
    assert af.min() >= -1e-12
    assert_allclose(af.sum(axis=0), 1, atol=1e-12, rtol=0)
    for name, abundance in [("reference", a), ("fcls", af)]:
        residual = y - m @ abundance
        assert_allclose(eda_arrays[f"residual_{name}"], residual, atol=1e-14)
        assert_allclose(eda_arrays[f"rmse_{name}_per_pixel"], np.sqrt(np.mean(residual ** 2, axis=0)), atol=1e-14)
    assert_allclose(eda_arrays["A_fcls_minus_A_ref"], af - a, atol=1e-14)
    assert_allclose(eda_arrays["armse_fcls_vs_reference_per_pixel"], np.sqrt(np.mean((af - a) ** 2, axis=0)), atol=1e-14)
    assert_array_equal(eda_arrays["dominant_reference_index_per_pixel"], a.argmax(axis=0))
    assert_array_equal(eda_arrays["purity_max_A_ref_per_pixel"], a.max(axis=0))
    terms = np.zeros_like(a)
    positive = a > 0
    terms[positive] = a[positive] * np.log(a[positive])
    assert_allclose(eda_arrays["entropy_normalized_A_ref_per_pixel"], -terms.sum(axis=0) / np.log(4), atol=1e-14)
    assert np.all(np.sum((y - m @ af) ** 2, axis=0) <= np.sum((y - m @ a) ** 2, axis=0) + 1e-10)
    # Convex simplex first-order oracle: all active gradients attain the minimum.
    gradient = 2 * m.T @ (m @ af - y)
    minimum = gradient.min(axis=0)
    active_gap = np.where(af > 1e-8, gradient - minimum, 0)
    assert active_gap.max() < 1e-7
    assert np.max(np.abs(np.sum(af * gradient, axis=0) - minimum)) < 1e-7


def test_eda_json_array_table_and_offline_link_coherence(clean, eda_arrays):
    import pandas as pd

    summary = strict_json(REPORTS / "eda_summary.json")
    assert summary["input_npz_sha256"] == sha256(PROCESSED / "jasper_clean.npz")
    assert summary["input_manifest_sha256"] == sha256(PROCESSED / "manifest.json")
    assert summary["material_order"] == MATERIALS
    assert_array_equal(summary["band_ids_original"], EXPECTED_BANDS)
    for key, array in eda_arrays.items():
        assert summary["array_contract"][key] == {"shape": list(array.shape), "dtype": str(array.dtype)}
    for key in ["pixel_ids", "row", "col", "valid_pixel_mask", "split_labels", "band_ids_original", "material_names"]:
        assert_array_equal(eda_arrays[key], clean[key])
    inv = summary["inventory"]
    assert inv["gt1_entries"] == 18 and inv["zero_entries"] == 418
    assert inv["all_zero_pixels"] == 0 and inv["valid_pixels"] == 10000
    assert inv["reference_asc_max_abs"] == float(np.max(np.abs(clean["A_ref"].sum(axis=0) - 1)))
    assert summary["pca"]["fit_count"] == 5800
    assert summary["pca"]["fit_pixel_ids_sha256"] == hashlib.sha256(clean["train_idx"].astype("<i8").tobytes()).hexdigest()
    assert summary["fcls"]["reference_used_in_solver"] is False
    assert summary["fcls"]["oracle_status"] == "passed"
    assert summary["proxies"]["adjacent_original_gap1_pair_count"] == 195
    assert summary["proxies"]["excluded_gap_pair_count"] == 2
    difference = np.diff(clean["Y"], axis=0)[np.diff(EXPECTED_BANDS) == 1]
    assert_allclose(eda_arrays["adjacent_band_difference_rms_proxy_per_pixel"], np.sqrt(np.mean(difference ** 2, axis=0)), atol=1e-14)
    frame = pd.read_csv(REPORTS / "tables/pca_explained_variance.csv")
    assert len(frame) == 198
    assert_allclose(frame["train_explained_variance"], eda_arrays["pca_explained_variance_train"], atol=1e-15, rtol=1e-12)
    oracle = pd.read_csv(REPORTS / "tables/fcls_slsqp_comparison.csv")
    assert_array_equal(oracle["pixel_id"], eda_arrays["slsqp_comparison_pixel_ids"])
    assert len(oracle) == summary["fcls"]["oracle_n"] == 128
    assert oracle["success"].all()
    html = (REPORTS / "eda_report.html").read_text(encoding="utf-8")
    links = re.findall(r'(?:href|src)=["\']([^"\']+)["\']', html)
    local = [link.split("#")[0] for link in links if not re.match(r"(?:https?:|mailto:|data:|#)", link)]
    assert any(link.endswith(".png") for link in local)
    for link in local:
        assert (REPORTS / link).is_file(), f"Broken offline link: {link}"

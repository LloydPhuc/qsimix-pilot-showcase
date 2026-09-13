"""Acceptance checks for test_plan.md; independent scalar circuit and synthetic inverse."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "validation"))
import jasper_pilot as pilot
from jasper_eda import fcls_exact
from jasper_pilot_data import load_npz, select_subset, write_json
from qsimix_residual_checks import ResidualCircuit
from residual_model import PairwiseModel, ResidualModel, fit_pairwise

CONFIG = json.loads((ROOT / "configs/jasper_pilot.json").read_text(encoding="utf-8"))


def test_subset_reproducibility_alignment_no_test_fitted_scale_or_label_selection():
    clean = load_npz(ROOT / "data/processed/jasper_ridge/jasper_clean.npz")
    actual = select_subset(clean, CONFIG)
    again = select_subset(clean, CONFIG)
    for key in actual:
        assert_array_equal(actual[key], again[key])
    bands, ids = actual["band_indices_clean"], actual["pixel_ids"]
    assert actual["Y"].shape == (32, 112)
    assert_allclose(actual["Y"] * actual["scale"], clean["Y"][np.ix_(bands, ids)], atol=1e-15)
    assert_allclose(actual["M"] * actual["scale"], clean["M"][bands], atol=1e-15)
    assert_array_equal(actual["A_ref"], clean["A_ref"][:, ids])
    assert_array_equal(actual["row"], ids % 100)
    assert_array_equal(actual["col"], ids // 100)
    for split in ("train", "val", "test"):
        assert set(ids[actual[f"{split}_idx"]]) <= set(clean[f"{split}_idx"])
    changed = {k: v.copy() for k, v in clean.items()}
    changed["Y"][:, clean["test_idx"]] *= 10
    changed["A_ref"] = np.roll(changed["A_ref"], 1, axis=0)
    modified = select_subset(changed, CONFIG)
    for key in ("pixel_ids", "band_ids_original", "scale", "M"):
        assert_array_equal(modified[key], actual[key])
    assert np.max(modified["Y"][:, modified["test_idx"]]) > 1  # no hidden clipping
    for split in ("train", "val"):
        assert_array_equal(modified["Y"][:, modified[f"{split}_idx"]], actual["Y"][:, actual[f"{split}_idx"]])


def test_batched_circuit_matches_independent_scalar_and_hard_anchor():
    rng = np.random.default_rng(17)
    m = rng.uniform(.1, .8, (5, 4))
    t0, theta = rng.normal(0, .2, 14), rng.normal(0, .8, 14)
    a = np.column_stack([np.eye(4), rng.dirichlet(np.ones(4), 3).T])
    model = ResidualModel(m, t0, theta)
    oracle = ResidualCircuit(4)
    expected = np.array([[oracle.q(p, band, theta) for p in a.T] for band in m])
    assert_allclose(model.q(a, theta), expected, atol=2e-14)
    assert_allclose(model.forward(np.eye(4)), m, atol=1e-14)
    assert_allclose(ResidualModel(m, t0).forward(a), m @ a, atol=1e-14)
    assert not model.theta0.flags.writeable


def test_complete_input_jacobian_matches_occurrence_shift_including_reference():
    rng = np.random.default_rng(22)
    m, t0, t = rng.uniform(.1, .8, (4, 4)), rng.normal(0, .2, 14), rng.normal(0, .7, 14)
    a = np.array([.11, .29, .37, .23])
    model, oracle = ResidualModel(m, t0, t), ResidualCircuit(4)
    expected = m + .03 * np.array([oracle.anchored_input_gradient(a, b, t)
                                    - oracle.anchored_input_gradient(a, b, t0) for b in m])
    assert_allclose(model.jacobian(a), expected, atol=2e-9, rtol=2e-7)
    # The witness would fail if the fixed reference were detached from abundance.
    omitted = m + .03 * np.array([oracle.anchored_input_gradient(a, b, t) for b in m])
    assert np.max(abs(expected - omitted)) > 1e-3


def test_parameter_finite_differences_against_shift_and_nonzero_initial_loss_gradient():
    rng = np.random.default_rng(31)
    m, t0 = rng.uniform(.1, .8, (3, 4)), rng.normal(0, .2, 14)
    a = rng.dirichlet(np.ones(4), 4).T
    model = ResidualModel(m, t0)
    shift_derivatives, difference_derivatives = [], []
    for direction in np.eye(14):
        shift_derivatives.append(.03 / 2 * (model.anchored(a, t0 + np.pi / 2 * direction)
                                             - model.anchored(a, t0 - np.pi / 2 * direction)))
        difference_derivatives.append(.03 / 2e-6 * (model.anchored(a, t0 + 1e-6 * direction)
                                                     - model.anchored(a, t0 - 1e-6 * direction)))
    assert_allclose(difference_derivatives, shift_derivatives, atol=2e-10, rtol=2e-6)
    # Synthetic target deliberately has a tangent component; not a universal trainability claim.
    target = model.forward(a) + shift_derivatives[0]
    analytic = 2 * np.mean((model.forward(a) - target) * shift_derivatives[0])
    assert abs(analytic) > 1e-8
    def loss(t):
        return np.mean((ResidualModel(m, t0, t).forward(a) - target) ** 2)
    direction = np.eye(14)[0]
    assert_allclose((loss(t0 + 1e-6 * direction) - loss(t0 - 1e-6 * direction)) / 2e-6,
                    analytic, atol=1e-11)


def test_pairwise_known_coefficients_and_vertices():
    rng = np.random.default_rng(4)
    m = rng.uniform(.1, .9, (12, 4))
    a = rng.dirichlet(np.ones(4), 48).T
    truth = np.array([.1, -.3, .7, .2, -.1, .5])
    model = PairwiseModel(m, truth)
    fitted = fit_pairwise(model.forward(a), m, a, 0.)
    assert_allclose(fitted, truth, atol=1e-12)
    assert_allclose(model.forward(np.eye(4)), m, atol=1e-14)


@pytest.mark.parametrize("nonlinear", [False, True])
def test_noiseless_synthetic_inverse_oracle_with_known_abundance(nonlinear):
    rng = np.random.default_rng(8)
    m, t0 = rng.uniform(.1, .9, (12, 4)), rng.normal(0, .2, 14)
    t = t0 + rng.normal(0, .2, 14) if nonlinear else t0
    model = ResidualModel(m, t0, t)
    truth = np.column_stack([np.eye(4), rng.dirichlet(np.ones(4), 2).T])
    y = model.forward(truth)
    fcls, _ = fcls_exact(y, m)
    estimates, log = pilot.inverse_model(model, y, fcls, {**CONFIG, "inverse_ftol": 1e-13})
    assert_allclose(estimates, truth, atol=2e-5)
    assert_allclose(model.forward(estimates), y, atol=1e-5)
    assert log["failed_starts"] == 0


def test_inverse_rejects_infeasible_success_and_reports_fallback(monkeypatch):
    monkeypatch.setattr(pilot, "minimize", lambda *args, **kwargs:
                        SimpleNamespace(x=np.array([2., 0, 0, 0]), success=True, message="fake success", nit=1))
    m = np.eye(4)
    model = PairwiseModel(m, np.zeros(6))
    truth = np.array([[.1], [.2], [.3], [.4]])
    estimated, log = pilot.inverse_model(model, truth, truth, CONFIG)
    assert_array_equal(estimated, truth)
    assert log["fallback_pixels"] == 1
    assert all(not start["feasible"] for start in log["pixels"][0]["starts"])


def test_validation_checkpoint_selection_includes_initial_and_excludes_test(monkeypatch):
    # Independent optimizer stub presents a worse val checkpoint after a good initialization.
    def optimizer(fun, x0, callback, **kwargs):
        candidate = x0 + .5
        callback(candidate)
        return SimpleNamespace(x=candidate, success=False, message="budget", nit=1, nfev=2)
    monkeypatch.setattr(pilot, "minimize", optimizer)
    rng = np.random.default_rng(7)
    m, a = rng.uniform(.1, .8, (8, 4)), rng.dirichlet(np.ones(4), 5).T
    t0, chosen, info = pilot.train_residual(m, m @ a + .01, a, m @ a, a, CONFIG, 1.)
    assert_array_equal(chosen, t0)
    assert info["selected_iteration"] == 0
    assert info["success"] is False


def test_stale_checkpoint_is_rejected(tmp_path):
    (tmp_path / "dependency.npz").write_bytes(b"changed")
    write_json(tmp_path / "metadata.json", {"subset_sha256": "old"})
    with pytest.raises(ValueError, match="Stale"):
        pilot._verify_artifact(tmp_path, "checkpoint.npz", "metadata.json", {"subset_sha256": "dependency.npz"})


def test_saved_results_metrics_and_no_label_inverse_inputs():
    output = ROOT / "reports/jasper_pilot"
    data = pilot._verify_subset(output, CONFIG)
    result = load_npz(output / "inverse.npz")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"),
                         parse_constant=lambda value: pytest.fail(f"Invalid JSON: {value}"))
    idx = data["test_idx"]
    assert_array_equal(result["test_pixel_ids"], data["pixel_ids"][idx])
    for name in ("fcls", "pairwise", "qsimix"):
        a, predicted = result[f"A_{name}"], result[f"Y_{name}"]
        recorded = summary["test_inverse"][name]
        assert_allclose(recorded["spectral_rmse_nominal"],
                        np.sqrt(np.mean((data["scale"] * (data["Y"][:, idx] - predicted)) ** 2)))
        assert_allclose(recorded["armse_vs_reference"], np.sqrt(np.mean((a - data["A_ref"][:, idx]) ** 2)))
        assert np.min(a) >= -1e-8
        assert_allclose(a.sum(axis=0), 1, atol=1e-8)
    log = json.loads((output / "training.json").read_text(encoding="utf-8"))
    assert log["selected_iteration"] == np.argmin([h["val_mse_model_scale"] for h in log["history"]])

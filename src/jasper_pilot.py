"""Stage-based exploratory pilot. Checkpoints make each task independently runnable."""
from __future__ import annotations

import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize

from jasper_eda import fcls_exact, spectral_angle
from jasper_pilot_data import load_npz, prepare_subset, sha256, write_json
from residual_model import PairwiseModel, ResidualModel, fit_pairwise

STAGES = ("subset", "baselines", "train", "inverse", "report")


def train_residual(M, Y_train, A_train, Y_val, A_val, config, scale):
    """Test arrays are deliberately absent from the training API."""
    started = time.perf_counter()
    theta0 = np.random.default_rng(config["seed"]).normal(0, .2, 14)
    model = ResidualModel(M, theta0, lam=config["lambda_nominal"] / scale)
    ref_train = model.anchored(A_train, theta0)
    linear_train = M @ A_train
    history = []
    best = {"theta": theta0.copy(), "val_mse": float("inf"), "iteration": 0}

    def prediction(theta, A):
        reference = ref_train if A is A_train else model.anchored(A, theta0)
        return M @ A + model.lam * (model.anchored(A, theta) - reference)

    def objective(theta):
        error = linear_train + model.lam * (model.anchored(A_train, theta) - ref_train) - Y_train
        return float(np.mean(error ** 2) + config["parameter_l2"] * np.mean((theta - theta0) ** 2))

    def record(theta):
        train_mse = float(np.mean((prediction(theta, A_train) - Y_train) ** 2))
        val_mse = float(np.mean((prediction(theta, A_val) - Y_val) ** 2))
        iteration = len(history)
        history.append({"iteration": iteration, "train_mse_model_scale": train_mse,
                        "val_mse_model_scale": val_mse, "regularized_loss": objective(theta)})
        if val_mse < best["val_mse"]:
            best.update(theta=theta.copy(), val_mse=val_mse, iteration=iteration)
        if iteration % 10 == 0:
            print(f"train iteration={iteration}, val RMSE nominal={scale*np.sqrt(val_mse):.7f}", flush=True)

    record(theta0)
    # CPU smoke implementation: finite differences, not claimed autodiff/Adam.
    result = minimize(objective, theta0.copy(), method="L-BFGS-B", callback=record,
                      options={"maxiter": config["train_maxiter"], "ftol": config["train_ftol"],
                               "gtol": 1e-8, "eps": config["difference_step"]})
    return theta0, best["theta"], {
        "optimizer": "L-BFGS-B, forward numerical differences; 40 iterations by default",
        "success": bool(result.success), "message": str(result.message), "iterations": int(result.nit),
        "function_evaluations": int(result.nfev), "selected_iteration": best["iteration"],
        "selection": "minimum validation forward MSE over accepted iterates, including initial LMM",
        "lambda_model_scale": model.lam, "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "theta_distance_l2": float(np.linalg.norm(best["theta"] - theta0))}


def inverse_model(model, Y, fcls, config):
    """Only observed spectra, fixed model, FCLS starts; never takes A_ref."""
    started = time.perf_counter()
    estimates, logs = [], []
    for pixel in range(Y.shape[1]):
        y = Y[:, pixel]

        def objective(a):
            error = model.forward(a[:, None])[:, 0] - y
            return float(.5 * np.mean(error ** 2))

        def jac(a):
            error = model.forward(a[:, None])[:, 0] - y
            return model.jacobian(a, config["difference_step"]).T @ error / y.size

        candidates, starts = [], []
        for label, initial in (("fcls", fcls[:, pixel]), ("uniform", np.full(4, .25))):
            # Explicit feasible fallbacks ensure a failed solver never deletes a pixel.
            candidates.append((objective(initial), initial.copy(), f"{label}_initial", None))
            result = minimize(objective, initial.copy(), jac=jac, method="SLSQP",
                              bounds=[(0., 1.)] * 4,
                              constraints={"type": "eq", "fun": lambda a: a.sum() - 1,
                                           "jac": lambda a: np.ones(4)},
                              options={"ftol": config["inverse_ftol"], "maxiter": config["inverse_maxiter"]})
            a = result.x
            feasible = bool(np.isfinite(a).all() and a.min() >= -1e-8 and a.max() <= 1 + 1e-8
                            and abs(a.sum() - 1) <= 1e-8)
            value = objective(a) if feasible else None
            starts.append({"start": label, "success": bool(result.success), "message": str(result.message),
                           "iterations": int(result.nit), "feasible": feasible,
                           "objective": value, "abundance": a.tolist() if np.isfinite(a).all() else None})
            if feasible and np.isfinite(value):
                candidates.append((value, a.copy(), label, bool(result.success)))
        best = min(candidates, key=lambda item: item[0])
        estimates.append(best[1])
        final_values = [item["objective"] for item in starts if item["objective"] is not None]
        logs.append({"local_pixel": pixel, "selected_start": best[2], "selected_solver_success": best[3],
                     "objective": best[0], "starts": starts,
                     "final_start_objective_gap": max(final_values) - min(final_values) if final_values else None})
    return np.array(estimates).T, {"elapsed_seconds": time.perf_counter() - started, "pixels": logs,
                                  "failed_starts": sum(not s["success"] for p in logs for s in p["starts"]),
                                  "fallback_pixels": sum(p["selected_solver_success"] is None for p in logs)}


def metrics(Y, predicted, A, reference, scale):
    residual = scale * (Y - predicted)
    energy, error = float(np.sum((scale * Y) ** 2)), float(np.sum(residual ** 2))
    sam = spectral_angle(Y, predicted)
    defined = np.isfinite(sam)
    return {"n_pixels": Y.shape[1], "spectral_rmse_nominal": float(np.sqrt(np.mean(residual ** 2))),
            "sre_db": float(10 * np.log10(energy / error)) if energy > 0 and error > 0 else None,
            "sre_status": "defined" if energy > 0 and error > 0 else "undefined_zero_energy",
            "sam_mean_radians": float(sam[defined].mean()) if defined.any() else None,
            "sam_defined_count": int(defined.sum()),
            "armse_vs_reference": float(np.sqrt(np.mean((A - reference) ** 2))),
            "per_material_armse_vs_reference": np.sqrt(np.mean((A - reference) ** 2, axis=1)).tolist(),
            "per_material_bias_vs_reference": np.mean(A - reference, axis=1).tolist(),
            "asc_max_abs": float(np.max(abs(A.sum(axis=0) - 1))),
            "anc_max_violation": float(max(0., -A.min())),
            "prediction_outside_model_0_1_count": int(((predicted < 0) | (predicted > 1)).sum())}


def _models(data, checkpoint):
    if not np.array_equal(data["M"], checkpoint["M"]):
        raise ValueError("Checkpoint M does not match subset")
    return {"pairwise": PairwiseModel(checkpoint["M"], checkpoint["pairwise_coefficients"]),
            "qsimix": ResidualModel(checkpoint["M"], checkpoint["theta0"], checkpoint["theta"],
                                    float(checkpoint["lambda_model_scale"]))}


def _verify_subset(output, config):
    import json
    meta = json.loads((output / "subset_manifest.json").read_text(encoding="utf-8"))
    if meta["config"] != config or meta["subset_sha256"] != sha256(output / "subset.npz"):
        raise ValueError("Subset/config changed: rerun subset then dependent stages")
    return load_npz(output / "subset.npz")


def _verify_artifact(output, name, metadata, dependencies):
    import json
    info = json.loads((output / metadata).read_text(encoding="utf-8"))
    for key, dependency in dependencies.items():
        if info[key] != sha256(output / dependency):
            raise ValueError(f"Stale {name}: rerun its stage after {dependency}")
    if info["artifact_sha256"] != sha256(output / name):
        raise ValueError(f"Modified artifact {name}: rerun its stage")
    return load_npz(output / name)


def run_stage(root, config, output, stage):
    if stage == "subset":
        return prepare_subset(root, config, output)
    data = _verify_subset(output, config)
    train, val = data["train_idx"], data["val_idx"]
    if stage == "baselines":
        start = time.perf_counter()
        a, diagnostics = fcls_exact(data["Y"], data["M"])
        if diagnostics["status"] != "passed":
            raise RuntimeError("FCLS KKT check failed")
        coefficients = fit_pairwise(data["Y"][:, train], data["M"], data["A_ref"][:, train], config["pairwise_l2"])
        np.savez_compressed(output / "baselines.npz", A_fcls=a, pairwise_coefficients=coefficients)
        info = {"subset_sha256": sha256(output / "subset.npz"),
                "artifact_sha256": sha256(output / "baselines.npz"),
                "fcls_kkt_max": diagnostics["kkt_max"], "elapsed_seconds": time.perf_counter() - start,
                "pairwise_parameters": 6, "pairwise_regularization": config["pairwise_l2"]}
        write_json(output / "baselines.json", info)
        return info
    baseline = _verify_artifact(output, "baselines.npz", "baselines.json", {"subset_sha256": "subset.npz"})
    if stage == "train":
        theta0, theta, info = train_residual(data["M"], data["Y"][:, train], data["A_ref"][:, train],
                                             data["Y"][:, val], data["A_ref"][:, val], config, float(data["scale"]))
        np.savez_compressed(output / "checkpoint.npz", M=data["M"], theta0=theta0, theta=theta,
                            lambda_model_scale=np.array(info["lambda_model_scale"]),
                            pairwise_coefficients=baseline["pairwise_coefficients"])
        info.update(subset_sha256=sha256(output / "subset.npz"), baseline_sha256=sha256(output / "baselines.npz"),
                    artifact_sha256=sha256(output / "checkpoint.npz"))
        write_json(output / "training.json", info)
        return info
    checkpoint = _verify_artifact(output, "checkpoint.npz", "training.json",
                                  {"subset_sha256": "subset.npz", "baseline_sha256": "baselines.npz"})
    models = _models(data, checkpoint)
    if stage == "inverse":
        # Only test spectra reach inverse; A_ref is read later by report.
        idx = data["test_idx"]
        arrays = {"test_pixel_ids": data["pixel_ids"][idx], "A_fcls": baseline["A_fcls"][:, idx]}
        info = {"subset_sha256": sha256(output / "subset.npz"), "checkpoint_sha256": sha256(output / "checkpoint.npz")}
        for name, model in models.items():
            print(f"inverse {name}: {idx.size} test pixels, FCLS + uniform starts", flush=True)
            arrays[f"A_{name}"], info[name] = inverse_model(model, data["Y"][:, idx], arrays["A_fcls"], config)
            arrays[f"Y_{name}"] = model.forward(arrays[f"A_{name}"])
        arrays["Y_fcls"] = data["M"] @ arrays["A_fcls"]
        np.savez_compressed(output / "inverse.npz", **arrays)
        info["artifact_sha256"] = sha256(output / "inverse.npz")
        write_json(output / "inverse.json", info)
        return info
    if stage != "report":
        raise ValueError(f"Unknown stage: {stage}")
    inverse = _verify_artifact(output, "inverse.npz", "inverse.json",
                               {"subset_sha256": "subset.npz", "checkpoint_sha256": "checkpoint.npz"})
    return make_report(data, models, inverse, output, config)


def make_report(data, models, inverse, output, config):
    import json
    import matplotlib.pyplot as plt

    scale = float(data["scale"])
    training = json.loads((output / "training.json").read_text(encoding="utf-8"))
    solver = json.loads((output / "inverse.json").read_text(encoding="utf-8"))
    subset = json.loads((output / "subset_manifest.json").read_text(encoding="utf-8"))
    idx = data["test_idx"]
    test = {name: metrics(data["Y"][:, idx], inverse[f"Y_{name}"], inverse[f"A_{name}"],
                          data["A_ref"][:, idx], scale) for name in ("fcls", "pairwise", "qsimix")}
    forward = {}
    for split in ("train", "val", "test"):
        ids = data[f"{split}_idx"]
        a, y = data["A_ref"][:, ids], data["Y"][:, ids]
        forward[split] = {"lmm_at_reference": float(scale * np.sqrt(np.mean((data["M"] @ a - y) ** 2)))}
        forward[split].update({name: float(scale * np.sqrt(np.mean((model.forward(a) - y) ** 2)))
                              for name, model in models.items()})
    summary = {"scope": "single-seed exploratory pilot, 32 selected bands by default; no physical-truth or advantage claim",
               "config": config, "scale": scale, "coverage": subset["coverage"],
               "forward_spectral_rmse_nominal": forward, "test_inverse": test,
               "training": {key: training[key] for key in ("success", "message", "iterations", "selected_iteration", "elapsed_seconds")},
               "solver": {name: {key: solver[name][key] for key in ("failed_starts", "fallback_pixels", "elapsed_seconds")}
                          for name in models},
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
               "hashes": {name: sha256(output / name) for name in ("subset.npz", "baselines.npz", "checkpoint.npz", "inverse.npz")}}
    write_json(output / "summary.json", summary)
    # Abundance plots show all 24 test pixels in original-ID order, not a fabricated contiguous crop.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    x = np.arange(idx.size)
    for material, ax in enumerate(axes.flat):
        ax.plot(x, data["A_ref"][material, idx], "k--", label="Reference estimates", linewidth=1.4)
        for name in test:
            ax.plot(x, inverse[f"A_{name}"][material], ".-", label=name, alpha=.8)
        ax.set_title(str(data["material_names"][material]))
        ax.set_ylim(-.03, 1.03)
        ax.set_xlabel("Test sample index (sorted original pixel IDs)")
        ax.set_ylabel("Abundance")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "abundance_comparison.png", dpi=150)
    plt.close(fig)
    lines = ["# Jasper Ridge — pilot unmixing nhỏ", "",
             "Chạy một seed, chọn mẫu trong split spatial cũ; đây là thử pipeline exploratory, chưa phải benchmark đầy đủ.", "",
             f"Dữ liệu: {data['M'].shape[0]} band, train/val/test={train_sizes(data)}; 4 vật liệu tree, water, dirt, road. "
             f"Scale c={scale:.6g} lấy từ toàn train gốc và known M; Y, M, lambda cùng chia c. "
             "RMSE dưới đây đã đổi về thang nominal của EDA. Không clip phổ hoặc xóa outlier.", "",
             "## Forward tại A_ref (RMSE phổ)", "",
             "Calibration dùng reference estimates ở train; validation chỉ chọn checkpoint. Forward test dưới đây là đánh giá sau freeze.", "",
             "| Split | LMM tại A_ref | Pairwise 6 tham số | QSiMix 14 tham số |", "|---|---:|---:|---:|"]
    for split, values in forward.items():
        lines.append(f"| {split} | {values['lmm_at_reference']:.7f} | {values['pairwise']:.7f} | {values['qsimix']:.7f} |")
    lines += ["", "## Inverse trên test", "", "| Model | RMSE phổ | aRMSE với reference | SAM (rad) | SRE (dB) |", "|---|---:|---:|---:|---:|"]
    for name, result in test.items():
        sam = f"{result['sam_mean_radians']:.7f}" if result['sam_mean_radians'] is not None else "undefined"
        sre = f"{result['sre_db']:.4f}" if result['sre_db'] is not None else "undefined"
        lines.append(f"| {name} | {result['spectral_rmse_nominal']:.7f} | {result['armse_vs_reference']:.7f} | {sam} | {sre} |")
    lines += ["", "| Material | FCLS aRMSE-ref | Pairwise aRMSE-ref | QSiMix aRMSE-ref |", "|---|---:|---:|---:|"]
    for i, name in enumerate(data["material_names"]):
        lines.append(f"| {name} | " + " | ".join(f"{test[m]['per_material_armse_vs_reference'][i]:.7f}" for m in test) + " |")
    q, f = test["qsimix"], test["fcls"]
    lines += ["", f"QSiMix so với FCLS: ΔRMSE phổ={q['spectral_rmse_nominal']-f['spectral_rmse_nominal']:+.7f}; "
              f"ΔaRMSE-reference={q['armse_vs_reference']-f['armse_vs_reference']:+.7f} (âm là giảm). "
              "Hai chỉ số đo hai điều khác nhau; reconstruction tốt hơn không xác nhận abundance vật lý đúng hơn.", "",
              "![Abundance của từng pixel test](abundance_comparison.png)", "",
              "## Coverage và trạng thái solver", "",
              "| Split | Số mixture (max A_ref < .95) | Dominant tree/water/dirt/road |", "|---|---:|---|"]
    for split, coverage in subset["coverage"].items():
        lines.append(f"| {split} | {coverage['mixtures_below_095']} | {coverage['dominant_reference_counts']} |")
    lines += ["", f"Training L-BFGS-B: success={training['success']}; {training['message']}. "
              f"{training['iterations']} iterations; chọn iteration {training['selected_iteration']} theo val forward MSE "
              f"(có LMM initialization); thời gian {training['elapsed_seconds']:.2f}s. "
              "Ngân sách chạm maxiter không được gọi là hội tụ.", ""]
    for name in models:
        info, result = solver[name], test[name]
        lines.append(f"- {name}: failed starts={info['failed_starts']}/{2*idx.size}, fallback pixels={info['fallback_pixels']}; "
                     f"ASC max={result['asc_max_abs']:.3g}, ANC violation={result['anc_max_violation']:.3g}; "
                     f"inverse {info['elapsed_seconds']:.2f}s; output entries ngoài [0,1]={result['prediction_outside_model_0_1_count']}.")
    lines += ["", "Mỗi pixel chạy FCLS và uniform starts; chọn objective thấp nhất trong các kết quả khả thi, "
              "kể cả initial fallback có ghi nhãn. `inverse.json` giữ success/message/abundance/objective từng start "
              "và gap giữa hai nghiệm. Đây không phải chứng nhận global optimum.", "",
              "## Giới hạn và bước nhỏ kế tiếp", "",
              "- Sampling không dùng nhãn: có thể thiếu một vật liệu trong test nhỏ; giữ nguyên split, không chọn lại seed để làm đẹp kết quả.",
              "- Chỉ đánh giá trên các band đã chọn; không so trực tiếp số này với RMSE toàn cảnh 198 band trong EDA.",
              "- Pairwise là residual có neo với hệ số có dấu, không phải GBM vật lý; regularization cố định, chưa tuning công bằng ngân sách.",
              "- Exact NumPy statevector và finite differences; chưa Adam/autodiff, multi-seed, no-RZZ retraining, certificate, shots hay hardware.",
              "- Residual còn lẫn brightness, variability, scale/reference mismatch; A_ref không phải tỷ lệ đo thực địa. Test cũ đã xuất hiện trong EDA.",
              "- Task tiếp theo trong test_plan.md: kiểm no-RZZ bằng rectangle và retrain cùng subset; chưa cần mở rộng full plan.", "",
              "Số liệu đầy đủ: [summary.json](summary.json). Dữ liệu nhỏ: [subset.npz](subset.npz). "
              "Checkpoint: [checkpoint.npz](checkpoint.npz). Abundance và tái dựng: [inverse.npz](inverse.npz).", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def train_sizes(data):
    return "/".join(str(data[f"{split}_idx"].size) for split in ("train", "val", "test"))

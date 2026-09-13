# Hợp đồng test pilot Jasper Ridge

Ngày: 13-09-2026. **TP-01–TP-06 đã hoàn tất và chạy thực tế.** TP-07–TP-08 là hai task nhỏ để tiếp tục khi cần, chưa thực hiện. Đây là pilot cơ bản, không thay thế full benchmark trong directive.

## Mục tiêu và căn cứ

Trả lời câu hỏi hẹp: với một phần nhỏ Jasper Ridge, calibration rồi frozen inverse có chạy được, và abundance/reconstruction thay đổi thế nào so với FCLS?

Người nhận task đọc [EDA §7–9](reports/jasper_ridge/eda_report.md), [quyết định dữ liệu](reports/jasper_ridge/research_decisions.md), [review 13-9 §5 và §7](reviews/13-9.md), [đặc tả hiện hành §3–4, §7–8](directive/qsimix_ultimate.md) và [config riêng pilot](configs/jasper_pilot.json).

EDA chi phối thiết kế: FCLS đã có KKT/oracle; A_ref là reference estimates; dirt/road gần hướng phổ; residual lẫn brightness/variability; Y max=1.0874; spatial split có distribution shift. Vì vậy báo cả spectral RMSE và aRMSE-reference, không gọi residual là scattering đo được.

## Hợp đồng chung

1. **Subset:** 64 train / 24 val / 24 test, uniform không hoàn lại bằng seed 20260913 trong từng split cũ; không dùng A_ref để chọn, không dùng buffer. Đây là tập con pixel, không phải crop liền nhau. Chọn 32 vị trí band bằng `linspace(0,197,32,dtype=int)`. Lưu original IDs; không chọn lại seed/band theo kết quả test.
2. **Scale:** `c=max(1,max(Y_full_train),max(M_known))` trên toàn train gốc, tất cả 198 band, trước sampling. Thực tế c=1.0874. Chia đồng bộ Y, M, lambda=.03 cho c; phase dùng `pi*M_model`. Không clip input/output, không hứa test luôn <=1. Báo RMSE nhân lại c về thang nominal EDA; abundance không đổi scale.
3. **Schema:** Y[B,N], M[B,4], A_ref[4,N], float64; material order tree/water/dirt/road. `train_idx/val_idx/test_idx` trong subset là **vị trí cột subset**; `pixel_ids` ánh xạ ra ID gốc. Một abundance vector dùng chung mọi band của pixel.
4. **Protocol:** calibration dùng A_ref train; checkpoint chọn bằng val forward MSE. Freeze M/theta/theta0/lambda trước inverse. API inverse không nhận A_ref; report mới so với test reference.
5. **Model:** A0-v1, E=4, L=2, 14 tham số chung pixel/band; RY/RZ reupload → chain RZZ → local RY; mean-local-Z, hard anchor, fixed reference. NumPy exact statevector. Reference vẫn được tính lại theo abundance trong inverse.
6. **Prototype:** L-BFGS-B numerical gradient thay Adam/autodiff để dùng runtime Jasper hiện có; tối đa 40 iterations, beta=1e-4, lambda cố định. Không tuning grid, shots, learned M, spatial loss, certificate hoặc quantum advantage.
7. **Baseline:** reuse `jasper_eda.fcls_exact`; classical anchored pairwise có 6 hệ số có dấu chung band/pixel, ridge train. Không gọi hệ số này là scattering GBM vật lý; chưa tuning công bằng toàn bộ ngân sách.
8. **Bảo toàn:** không ghi đè raw, clean NPZ, EDA hoặc validation lịch sử. Hash liên kết subset → baseline → checkpoint → inverse; stage sau từ chối artifact cũ/đã bị sửa. Output mới nằm trong `reports/jasper_pilot/`.

## Cách giao việc

Mỗi prompt chỉ cần: **“Thực hiện/kiểm tra TP-xx trong test_plan.md; chỉ sửa write set của task; báo acceptance và artifact bàn giao.”** Nếu prerequisite thiếu, báo đúng chỗ thiếu, không tự triển khai cả chuỗi. Không cần giao nhiều agent.

Luồng: `TP-01 → TP-02 → TP-03 → TP-04 → TP-05 → TP-06`; sau đó tùy chọn TP-07 → TP-08. PASS nghĩa là đạt hợp đồng kỹ thuật, **không bắt buộc QSiMix thắng baseline**. Kết quả âm vẫn bàn giao.

### TP-01 — Tạo dữ liệu nhỏ [DONE]

- **Input:** clean NPZ/manifest trong `data/processed/jasper_ridge/`, config pilot.
- **Write set:** `src/jasper_pilot_data.py`, `configs/jasper_pilot.json`; output `subset.npz`, `subset_manifest.json`.
- **Việc làm/output:** sample, scale và giữ metadata/hashes qua `prepare_subset(root,config,output)`; đếm mixture/coverage theo split nhưng không resample theo nhãn.
- **Acceptance:** Y=(32,112), M=(32,4), A_ref=(4,112); disjoint, không buffer/invalid; khớp clean và mapping; scale/IDs không đổi khi thay test spectra/nhãn; train có mixture; input sai phải fail rõ.
- **Lệnh:** `.venv-jasper/Scripts/python.exe scripts/run_jasper_pilot.py --stage subset`
- **Bàn giao:** 37 mixture train (max A_ref<.95); dominant train `[19,33,12,0]`. **Thiếu road-dominant train**, không che hoặc chọn lại seed.

### TP-02 — Baseline trên đúng subset [DONE]

- **Prerequisite:** TP-01.
- **Write set:** phần pairwise trong `src/residual_model.py`, stage `baselines` trong `src/jasper_pilot.py`; output `baselines.npz/json`.
- **Việc làm/output:** FCLS tính lại trên 32 band, không cắt A_fcls EDA 198 band để thay thế; pairwise fit train. Lưu A_fcls[4,112], coefficients[6], KKT/time.
- **Acceptance:** FCLS feasible, KKT<=1e-7; pairwise neo đúng đỉnh, synthetic noiseless recover coefficients khi ridge=0; test labels không vào fit.
- **Lệnh:** `.venv-jasper/Scripts/python.exe scripts/run_jasper_pilot.py --stage baselines`
- **Bàn giao:** FCLS max KKT=1.92e-15.

### TP-03 — Decoder A0 tối thiểu [DONE]

- **Prerequisite:** TP-01; đối chiếu scalar oracle `validation/qsimix_residual_checks.py`, không sửa oracle.
- **Write set:** `ResidualModel` trong `src/residual_model.py`; tests mạch trong `tests/test_jasper_pilot.py`.
- **API/output:** `ResidualModel(M,theta0,theta,lam)`, `q(A,theta)`, `forward(A)->[B,N]`, `jacobian(a)->[B,4]`.
- **Acceptance:** batch q khớp scalar <=2e-14; LMM initialization/vertices <=1e-14; parameter finite differences khớp parameter-shift; abundance Jacobian khớp occurrence-shift **cả reference và anchor**; có synthetic witness gradient loss khác 0.
- **Lệnh:** `.venv-jasper/Scripts/python.exe -m pytest tests/test_jasper_pilot.py -q -k "circuit or jacobian or parameter"`
- **Giới hạn:** Jacobian central differences h=1e-6, không gọi là autodiff hoặc simultaneous input parameter-shift. Cache vertices của model chỉ dùng khi parameters đã freeze.

### TP-04 — Calibration và checkpoint [DONE]

- **Prerequisite:** TP-02, TP-03.
- **Write set:** `train_residual` và stage `train` trong `src/jasper_pilot.py`; output `checkpoint.npz`, `training.json`.
- **API/input:** `train_residual(M,Y_train,A_train,Y_val,A_val,config,scale)`; không có test trong API.
- **Việc làm:** theta0 seeded N(0,.2²), theta=theta0; loss mean spectral error² + beta*mean((theta-theta0)²). Chọn minimum val MSE qua accepted iterates, **có cả initial LMM**.
- **Output/acceptance:** checkpoint finite lưu M/theta0/theta/lambda/baseline coefficients; history, selected iteration, status/message/nfev/time, hashes. Không train reference, không che maxiter/failure; cho phép chọn theta0 nếu val không cải thiện.
- **Lệnh:** `.venv-jasper/Scripts/python.exe scripts/run_jasper_pilot.py --stage train`
- **Bàn giao:** hội tụ sau 27 iterations; chọn **iteration 2**. Forward val nominal RMSE 0.0480227 → 0.0454305.

### TP-05 — Frozen inverse 24 pixel [DONE]

- **Prerequisite:** TP-04.
- **Write set:** `inverse_model`, stage `inverse` trong `src/jasper_pilot.py`; output `inverse.npz/json`.
- **API/input:** `inverse_model(model,Y,fcls,config)`; không A_ref, dùng checkpoint đã freeze.
- **Việc làm:** SLSQP .5*mean((F(a)-y)²), bounds [0,1], sum(a)=1; starts FCLS/uniform cho cả nonlinear models. Chọn lowest feasible objective, kể cả initial fallback có nhãn; không clip/normalize nghiệm.
- **Output:** original test IDs, A/Y_hat từng model; từng start có abundance/objective/success/message/iterations/feasible, gap objective, selected start và fallback count.
- **Acceptance:** ANC/ASC<=1e-8; synthetic noiseless recover abundance <=2e-5; success nhưng infeasible phải bị loại; không chọn nghiệm theo reference.
- **Lệnh:** `.venv-jasper/Scripts/python.exe scripts/run_jasper_pilot.py --stage inverse`
- **Bàn giao:** 0/48 failed starts mỗi model; QSiMix giữ 1 initial candidate vì objective thấp nhất, ghi rõ fallback. Có 3 reconstructed entries QSiMix ngoài [0,1], không clip; đây là hạn chế output-domain.

### TP-06 — Nghiệm thu và báo cáo [DONE]

- **Prerequisite:** TP-01–TP-05.
- **Write set:** stage `report`, `scripts/run_jasper_pilot.py`, `tests/test_jasper_pilot.py`, `README.md`, `test_plan.md`; output report/summary/figure/QA.
- **Việc làm/output:** forward train/val/test tại reference; inverse test spectral RMSE nominal/SRE/SAM rad/aRMSE/bias từng material, feasibility/failures/coverage/time. Vẽ từng sample theo original-ID order, không giả spatial crop.
- **Acceptance:** report khớp arrays/JSON; JSON không NaN/Inf; artifact đúng nguồn; regression EDA PASS; hình đọc được; không gọi aRMSE-ref là physical error hoặc pilot là full benchmark.
- **Lệnh report:** `.venv-jasper/Scripts/python.exe scripts/run_jasper_pilot.py --stage report`
- **Lệnh QA:** `.venv-jasper/Scripts/python.exe -m pytest tests/test_jasper_pilot.py tests/test_jasper_pipeline.py -q`
- **Bàn giao:** [report](reports/jasper_pilot/report.md), [summary](reports/jasper_pilot/summary.json), [QA](reports/jasper_pilot/qa.md). **47 PASS**: 11 test pilot + 36 test pipeline cũ.

### TP-07 — Rectangle no-RZZ [TODO, tùy chọn tiếp theo]

- **Prerequisite:** TP-03, TP-06; review 13-9 §7.
- **Write set dự kiến:** flag no-RZZ riêng trong `src/residual_model.py`, tests rectangle; `reports/jasper_pilot_no_rzz/rectangle.json`. Không ghi đè run đã có.
- **Việc làm/output:** bỏ RZZ ở **cả learned/reference**; rectangle a0=(.25,...), u=e1-e2, v=e3-e4, h=.1, fixed seeds như review; giữ encoding/readout/anchor.
- **Acceptance:** no-RZZ rectangle xấp xỉ 0, full A0 có witness khác 0; không gọi đây là accuracy hoặc entanglement advantage.
- **Điểm dừng:** chỉ regression/witness, chưa train hay inverse trong task này.

### TP-08 — Retrain no-RZZ cùng subset [TODO, chỉ sau TP-07]

- **Prerequisite:** TP-07; reuse đúng `subset.npz` TP-01.
- **Write set dự kiến:** config/runner no-RZZ nhỏ và `reports/jasper_pilot_no_rzz/`; không đổi test IDs hoặc pilot gốc.
- **Việc làm/output:** 8 local-RY parameters, không tính RZZ giả vào regularizer; init reference consistent, cùng budget/val rule/lambda/scale và hai inverse starts. Retrain từ đầu, không xóa RZZ sau train rồi gọi là baseline.
- **Acceptance:** bảng forward/inverse bốn model cùng diagnostics; báo đúng kể cả kết quả xấu; một seed chỉ exploratory.
- **Điểm dừng:** chưa multi-seed, 198 band, đổi split, full synthetic suite, Fourier/MLP tuning, certificate/hardware. Chỉ lập hợp đồng những việc đó nếu người dùng cần sau.

## Kết quả pilot đã chạy

| Model | Test spectral RMSE | aRMSE-reference |
|---|---:|---:|
| FCLS | 0.0489745 | 0.1014342 |
| Anchored pairwise | 0.0431614 | 0.1533348 |
| QSiMix-Residual | 0.0469697 | 0.0972717 |

QSiMix cải thiện nhẹ hai chỉ số so với FCLS trong đúng sample này; pairwise fit phổ tốt hơn nhưng abundance lệch reference nhiều hơn. Train thiếu road-dominant, val/test thiếu water-dominant. Chưa multi-seed hoặc abundance đo độc lập. Không so trực tiếp với RMSE toàn cảnh 198 band của EDA.

Chạy lại pilot TP-01–TP-06: `.venv-jasper/Scripts/python.exe scripts/run_jasper_pilot.py --stage all`; không chạy hai task TODO. Dùng `--output reports/ten_run_khac` để giữ bản kết quả; đổi config thì chạy lại stage phụ thuộc theo thứ tự.

Runtime `.venv-jasper`/`requirements-jasper.txt` hiện có, không cần Torch/PennyLane/mạng. Runner theo cách import trực tiếp `src/` của scripts EDA; không dùng `src/__init__.py` vì scaffold đó import `src.config` chưa tồn tại. `configs/default.yaml` vẫn giữ full specification, không bị đổi thành thông số smoke test.

# QSiMix — Quantum Simplex Mixing Model

> **⚠️ Exploratory Pilot Only — Not a Published Result**
> This repository contains an **early exploratory pilot** (TP-01–TP-06) on a small Jasper Ridge subset (32 bands, 112 pixels, 1 seed). It demonstrates a runnable end-to-end pipeline (preprocess → EDA → calibration → frozen inverse → tests) but **does not constitute a benchmark, peer-reviewed result, or quantum advantage claim**.
>
> Core methodology (QSiMix-Residual/A0-v1: proofs, approximation limits, benchmark design, ablation requirements) is documented in `directive/qsimix_ultimate.md` — **not included in this public repo**. See [Status](#trạng-thái-triển-khai) below.

**QSiMix-Residual là kiến trúc chủ đạo:** mạch lượng tử học phần phi tuyến cộng vào trộn tuyến tính \(Ma\), có neo tại vật liệu nguyên chất và khởi tạo đúng LMM. Phần cổ điển xử lý ràng buộc và tối ưu.

| Tên | Vai trò |
|---|---|
| **QSiMix-Residual** | Model chính: calibrated quantum nonlinear decoder, sau đó inverse abundance |
| QSiMix-GBM | Đối chứng biểu diễn GBM chính xác ở ideal expectation |
| QSiMix-Bernstein | Đối chứng đa thức và nền tảng hỗ trợ |

## Tài liệu

- **Bắt đầu tại [directive/README.md](directive/README.md)** để nạp pipeline và biết thứ tự đọc cho agent.
- [Mục lục archive](archive/README.md) chứa nền tảng, snapshot và khảo sát lịch sử; không phải đặc tả mặc định.
- [Plan chính QSiMix-Residual: pipeline, chứng minh, benchmark và references](directive/qsimix_ultimate.md).
- [Phản biện chuyên sâu Residual: proof, phản ví dụ và cách kiểm nghiệm](reviews/QSiMix_Residual_Deep_Review.md).
- [Sổ claim Residual: câu được phép viết, câu bị bác bỏ và câu còn mở](reviews/QSiMix_Residual_Claim_Register.md).
- [Bản Residual lưu nguyên trước phản biện chuyên sâu](archive/qsimix_residual_pre_deep_review.md).
- [Bản nền tảng GBM được lưu trước khi đổi hướng](archive/qsimix_gbm_foundation.md).
- [Phản biện phiên bản lịch sử IESA](<archive/phản biện 11-9.md>).
- [Methodology review của bản GBM](reviews/Methodology_Review_QSiMix_Ultimate.md), [EIC review của bản GBM](reviews/EIC_Review_QSiMix_Ultimate.md).
- [Phân loại và đánh giá tính mới QSiMix-GBM](reviews/QSiMix_GBM_Novelty_Assessment.md).

Protocol chính có endmembers biết trước và abundance labels cho calibration; test inverse dùng phổ quan sát. Đây chưa phải blind/fully unsupervised unmixing. Mạch quantum giữ vai trò học nonlinear residual; tính hữu ích so với classical residual và quantum advantage còn cần đánh giá. Pilot chain nông có thể mô phỏng bằng miền ảnh hưởng cục bộ.

## Kiểm chứng

Chạy từ thư mục unmixing:

~~~powershell
python validation/qsimix_residual_checks.py
python validation/qsimix_residual_deep_checks.py
~~~

[Kết quả Residual](validation/qsimix_residual_results.json): PASS các đẳng thức khởi tạo/neo, parameter-shift và abundance Jacobian trên E=3,4,6. Đây là kiểm tra đại số/statevector, chưa phải training benchmark.

[Kết quả deep audit](validation/qsimix_residual_deep_results.json): kiểm tra thêm circuit bằng dense matrices độc lập, exact local-cone evaluation/gradient, Fourier reconstruction, phản ví dụ vật liệu vắng mặt và covariance do dùng chung noisy anchors. Các kiểm tra số hỗ trợ proof review, không phải formal certificate hoặc quantum-advantage benchmark.

Các script dưới đây phục vụ bản GBM/Bernstein được lưu:

~~~powershell
python validation/qsimix_ultimate_checks.py
python validation/qsimix_ultimate_extended.py
~~~

[Kết quả GBM lịch sử](validation/qsimix_ultimate_results.txt) có thí nghiệm inverse tổng hợp dùng solver cổ điển; không phải kết quả inverse QSiMix-Residual.

## Trạng thái triển khai

[configs/default.yaml](configs/default.yaml) đã chuyển sang **đặc tả Residual**, E=4, L=2, chain, 14 tham số, mean-local-Z và reference subtraction. [Cấu hình legacy](configs/legacy_rzz_ry.yaml) giữ riêng.

Đã có **pilot Jasper nhỏ chạy được**: [test_plan.md](test_plan.md) tổ chức thành các hợp đồng TP-01–TP-08, có input/output, write set, acceptance và lệnh chạy từng task. TP-01–TP-06 đã hoàn tất: 64/24/24 pixel train/val/test, 32 band, FCLS + anchored pairwise + QSiMix-Residual, calibration rồi frozen inverse. [Báo cáo kết quả](reports/jasper_pilot/report.md), [QA 47 tests PASS](reports/jasper_pilot/qa.md).

```powershell
.venv-jasper/Scripts/python.exe scripts/run_jasper_pilot.py --stage all
```

Dùng `--stage subset|baselines|train|inverse|report` để chạy riêng từng bước (chọn một tên), và `--output reports/ten_run_khac` để giữ bản output hiện tại. Thông số pilot ở [configs/jasper_pilot.json](configs/jasper_pilot.json); không thay cấu hình đặc tả full benchmark. Pilot dùng NumPy/SciPy exact statevector, L-BFGS-B numerical gradient và SLSQP; chưa phải full training framework hoặc bằng chứng quantum advantage. A_ref là reference estimates, không phải abundance đo thực địa.

Full benchmark và các module triển khai chính theo directive chưa hoàn tất; `src/config.py` được import bởi scaffold cũ vẫn chưa tồn tại. Runner pilot dùng cách import trực tiếp từ `src/` như scripts EDA hiện có; các script validation tiếp tục chạy độc lập. Ngân sách full benchmark trong plan/config vẫn là đề xuất thực nghiệm.


# Jasper Ridge: bàn giao tiền xử lý và EDA

Đọc [kế hoạch điều phối](../../plan.md) để biết hợp đồng dữ liệu, nhiệm vụ từng agent và tiêu chuẩn nghiệm thu. [Quyết định dựa trên tài liệu](research_decisions.md) phân biệt các bước trong PDF với bước phù hợp cho file local; [source audit](source_audit.md) ghi bằng chứng từ MAT/FIG và SHA256.

Đã hoàn tất: 14 hình, 16 bảng CSV, báo cáo MD/HTML và bộ dữ liệu NPZ; **36 kiểm thử PASS**. Các kết quả dùng convention nominal `/5000` có khai báo, vẫn cần xác minh upstream trước khi coi là benchmark vật lý đã hiệu chuẩn.

## Chạy lại

Từ thư mục gốc repository, dùng CPython 3.12 trên Windows. Không dùng mặc định `python` nếu nó trỏ tới MSYS thiếu matplotlib. Môi trường `.venv-jasper` đã được tạo riêng cho công việc này.

```powershell
# Chỉ cần khi thiết lập một máy/môi trường mới:
py -3.12 -m venv .venv-jasper
.venv-jasper/Scripts/python.exe -m pip install -r requirements-jasper-lock.txt

# Tiền xử lý, EDA và kiểm thử:
.venv-jasper/Scripts/python.exe scripts/prepare_jasper.py
.venv-jasper/Scripts/python.exe scripts/eda_jasper.py
.venv-jasper/Scripts/python.exe -m pytest tests/test_jasper_pipeline.py -q --basetemp=tmp/jasper-qa-20260913

# Audit raw chỉ đọc, JSON in stdout:
.venv-jasper/Scripts/python.exe scripts/audit_jasper_source.py
```

Các CLI thêm đường dẫn module `src` trực tiếp để hoạt động độc lập với scaffold quantum hiện chưa đầy đủ. Môi trường tạo bằng uv có thể chưa có pip; trường hợp đó dùng `uv --cache-dir tmp/uv-cache pip install --python .venv-jasper/Scripts/python.exe -r requirements-jasper-lock.txt`. Cài dependency cần truy cập package registry, nhưng chạy pipeline đã cài xong không cần mạng.

`--basetemp` dành riêng cho fixture kiểm thử và pytest có thể dọn nội dung thư mục đó khi chạy lại; không đặt dữ liệu cá nhân vào đó. Flag này tránh lỗi quyền ở thư mục tạm mặc định Windows. Lần nghiệm thu: 36 passed in 5.57s, exit code 0.

## Artifact

| Artifact | Dùng để làm gì |
|---|---|
| `data/processed/jasper_ridge/jasper_clean.npz` | Y/M/A_ref, original IDs, pixel coordinates, mask và spatial splits |
| `data/processed/jasper_ridge/manifest.json` | Nguồn, hash, phép scale, orientation, policy, phiên bản runtime |
| [EDA report](eda_report.md) / [HTML offline](eda_report.html) | Phân tích định lượng và hình minh họa |
| `eda_summary.json`, `eda_arrays.npz` | Số liệu và mảng tính toán để tái sử dụng/kiểm tra |
| `tables/`, `figures/` | Bảng đầy đủ và ảnh PNG; giữ cùng thư mục để HTML hiển thị offline |
| [Independent QA](qa_review.md) | Kết quả kiểm tra độc lập và giới hạn còn lại |

Không sửa các raw files trong `data`. Không cần copy dữ liệu sang môi trường khác hoặc cài PennyLane/PyTorch để chạy EDA này.

## Hợp đồng khi dùng tiếp

`Y.shape=(198,10000)`, `M.shape=(198,4)`, `A_ref.shape=(4,10000)`. Thứ tự vật liệu là tree/water/dirt/road. Nominal convention là Y_raw/5000, M giữ nguyên; không clip Y>1, không chia lại M. Original band IDs là 1-based và có gaps. Pixel IDs là cột nguồn, 0-based. Xem schema đầy đủ trong plan và manifest.

M/A là reference benchmark; nguồn upstream, physical calibration và tương ứng Y-M-A chưa được chứng minh hoàn toàn từ các file hiện có. Các phép phân tích chung được báo có điều kiện theo convention đã công bố. Known-M dùng chung một library reference, không có cam kết M được trích chỉ từ train.

Các split đã được khảo sát trong EDA; chúng phù hợp phát triển thí nghiệm nhưng không còn là một test set hoàn toàn chưa được quan sát. PCA/thresholds fitted vẫn chỉ dùng train. Không dùng các phát hiện trên test để sửa split rồi gọi kết quả là đánh giá độc lập. Nếu huấn luyện QSiMix sau này, cần chốt protocol reference-label, scale/miền encoding/lambda, và thiết kế đánh giá confirmatory riêng.

# Independent QA — Jasper Ridge

Ngày: 2026-09-13. Agent D. Phạm vi sửa: `tests/test_jasper_pipeline.py` và báo cáo này; không sửa implementation, config, raw hoặc scaffold `src/__init__.py`.

## Trạng thái thực chạy

**Hoàn tất QA phần mềm: 36 passed trong 5.57 s, exit code 0, không warnings.** Parent trực tiếp chạy bộ tests độc lập do D viết và cung cấp kết quả cuối; D cập nhật báo cáo theo kết quả đó, không chạy lặp lại.

Lệnh cuối parent đã thực thi từ repo root:

```powershell
.venv-jasper/Scripts/python.exe -m pytest tests/test_jasper_pipeline.py -q --basetemp=tmp/jasper-qa-20260913
```

Đã đọc toàn bộ `plan.md`, phần giới hạn bổ sung và `source_audit.md`. Kiểm numerical theo contract, không thực hiện lại nghiên cứu nguồn hoặc kiểm hình của parent.

Runtime: Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2, pandas 3.0.5, pytest 9.1.1; môi trường `.venv-jasper`.

- Lần raw/FIG ban đầu: **9 passed, 4 deselected**, 1.48 s. Một cảnh báo pytest về iterator parametrize đã được sửa bằng list.
- Lần suite đầu tiên sau khi có artifacts: **25 passed, 1 failed, 10 errors**, 5.40 s. 10 lỗi setup là `PermissionError` ở thư mục temporary mặc định Windows `pytest-of-admin`; các test đó chưa thực thi. Một failure là link offline `qa_review.md` chưa tồn tại khi báo cáo này chưa được tạo, không phải sai số tính toán B/C.
- Yêu cầu chạy lại ngoài sandbox của D bị hủy khi parent nhận thực thi; không có kết quả pytest từ yêu cầu đó và không tính là pass.
- Lần cuối do parent thực thi với `--basetemp` trong workspace: **36 passed**, 5.57 s, exit code 0, không warnings. Toàn bộ 10 test trước bị lỗi setup và test link offline đã được thực thi thành công. Các lần lỗi ở trên được giữ làm lịch sử, không phải trạng thái hiện tại.

Không còn failure hoặc concern về numerical implementation chưa giải quyết trong phạm vi 36 tests này. Các giới hạn nguồn dữ liệu, calibration, alignment và khả năng suy luận khoa học bên dưới vẫn còn hiệu lực; kết quả pytest không tự động xác nhận chúng.

## Các oracle độc lập đã viết

| Rủi ro | Kiểm tra |
|---|---|
| Raw đổi byte hoặc sai config hash | Pin SHA256 cả 4 MAT/FIG từ raw đo trước xử lý; kiểm cả config, manifest và hash sau hai lần chạy preprocessing trên bản sao tạm |
| Drop/scale lần hai, clip hoặc đổi reference | Exact 198 retained IDs bằng set subtraction độc lập; Y bằng raw float64/5000 từng giá trị; M/A_ref bằng raw từng giá trị; dtype, shape, Unicode names, roundtrip scale |
| Orientation | FIG A và M khớp đúng nhãn tree/water/dirt/road; fixture 3×5 có grid chỉ số literal và pixel (row=1,col=3) phải bằng id=10, không chỉ kiểm nghịch đảo hai phép reshape |
| Mask/split | Giữ zero đơn lẻ, dark pixels và duplicates; mask nonfinite/negative/all-zero; inject all-zero trong RAM ở cả train/buffer/val/test; không mất cột hoặc đổi M/A; union/disjoint/unique ID và hai cột buffer |
| Validation | Fail-fast thiếu Y, transpose, dtype float, wrong bands, permuted names, negative M, NaN A và ASC sai; không xuất NPZ khi validation fail |
| Reproducibility | Chạy `prepare_dataset` hai lần trong root tạm, so sánh mọi key/value/dtype với nhau và artifact live; kiểm raw trước/sau |
| PCA leakage | Mean/covariance từ train độc lập bằng NumPy covariance, variance bằng SVD; reconstruct covariance từ components; perturb held-out mean lớn và NaN held-out phải không đổi fit; kiểm export fit IDs và transform Y/M |
| FCLS | Tổng hợp cả 15 supports, vertices/interior; counterexample clip+normalize; permutation equivariance; noisy full-rank và rank-deficient đối chiếu SLSQP; sample real seed riêng và edge IDs; feasibility, objective và simplex first-order optimality toàn cảnh |
| EDA coherence | JSON chuẩn không NaN/Infinity/overflow; input hashes; array dtype/shape contract; names/IDs; residual/RMSE/aRMSE/entropy/purity/dominant; chỉ 195 adjacent original-band pairs; PCA và SLSQP CSV; link HTML offline |
| Robust flags | Zero MAD phải null threshold/status undefined; median/MAD fit train-only, không đổi khi perturb held-out |

## Giới hạn khoa học và coverage

Pass kiểm phần mềm chỉ chứng minh thực thi các convention đã khai báo. `Y/5000` là nominal convention dựa trên scalar metadata, chưa chứng nhận calibration vật lý hoặc cùng scale với M. Không dùng labels tối ưu divisor. FIG xác minh A F-order và giá trị M, không chứng minh Y-A pairing hoặc band/wavelength alignment M-Y. Fixture non-square xác minh mapping phần mềm, không bổ sung bằng chứng nguồn gốc cho Y.

Giữ 18 entries Y>1, không có all-zero pixel ở raw thật, và A sum max-error xấp xỉ 1.9206858326015208e-14. Miền pilot [0,1] còn cần quyết định adapter/domain riêng trước training. Reference là benchmark reference, không phải chứng nhận measured ground truth; split exploratory đã được nhìn trong EDA, không phải test chưa quan sát cho confirmatory claims.

Synthetic FCLS có hệ full-rank và rank-deficient nhưng không chứng minh độ bền cho mọi condition number/scale ngoài Jasper. SLSQP oracle real kiểm sample độc lập; first-order optimality kiểm toàn cảnh. PCA tests không audit các pipeline training tương lai. Test masks negative/nonfinite qua helper; đường MAT thật chấp nhận uint16 nên các giá trị ấy bị chặn ở dtype validation.

Parent chịu trách nhiệm review 14 figures, captions và nguồn nghiên cứu. QA chỉ kiểm tồn tại/link và numerical coherence của artifacts, không tuyên bố đã xem hình. Không chạy toàn repository vì scaffold `src.config` lỗi preexisting; tests import trực tiếp `jasper_preprocess`/`jasper_eda` qua `repo/src`.

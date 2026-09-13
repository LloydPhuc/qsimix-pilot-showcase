# Nghiệm thu pilot 13-09-2026

Đã chạy từng stage subset → baselines → train → inverse → report bằng runtime `.venv-jasper` hiện có.

```powershell
.venv-jasper/Scripts/python.exe -m pytest tests/test_jasper_pilot.py tests/test_jasper_pipeline.py -q
```

Kết quả: **47 passed** trong 6.37 giây: 11 test pilot và 36 test pipeline Jasper cũ. Có một `PytestCacheWarning` do quyền ghi cache Windows; không có test lỗi. Lần đầu 11 fixtures bị sandbox chặn thư mục tạm Windows; chạy lại với quyền đã được công cụ cho phép thì tất cả PASS.

Bằng chứng nghiệm thu:

- Subset khớp clean theo pixel/band IDs; scale và selection không đổi khi thay phổ test/hoán vị nhãn. Test vượt miền vẫn được giữ, không clip.
- Batch statevector khớp scalar oracle; lần đối chiếu trực tiếp max error 2.22e-16. Test hard anchor/LMM initialization và gradient tham số bằng parameter-shift đều đạt.
- Abundance Jacobian khớp scalar occurrence-shift, bao gồm reference/vertices. Witness cho thấy bỏ reference derivative sẽ sai.
- Synthetic noiseless LMM/QSiMix frozen inverse phục hồi abundance trong tolerance 2e-5; đây là recovery trên dữ liệu sinh bởi decoder, không phải accuracy vật lý của Jasper.
- Fake solver success nhưng infeasible bị loại, có fallback rõ. Validation selection có LMM initialization; API train không nhận test.
- Stale checkpoint bị reject; RMSE/aRMSE summary được tính lại trực tiếp từ NPZ.
- 36 regression tests kiểm raw hashes, preprocessing, split, FCLS/oracle và artifacts EDA cũ tiếp tục PASS.
- Đã mở kiểm tra `abundance_comparison.png`: đủ bốn vật liệu, trục chung, legend và nhãn đọc được, không cắt nội dung.

Giới hạn: một seed, 32 band, 24 test pixels; A_ref là reference estimates; train thiếu road-dominant. QSiMix có 3 entries phổ tái dựng ngoài [0,1], không clip. KKT FCLS không phải certificate nonlinear inverse; hai starts không bảo đảm global optimum. Chưa thực hiện TP-07/TP-08 no-RZZ.

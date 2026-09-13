# Audit nguồn Jasper Ridge — chỉ đọc dữ liệu

Ngày kiểm tra: 2026-09-13, Asia/Saigon. Workspace: `C:/Users/admin/Tai_lieu/detaikhoahoc/QCNN/unmixing`.

## Kết luận có thể sử dụng

Dữ liệu thực tế hỗ trợ bố cục canonical `Y=(198,10000)`, `M=(198,4)`, `A=(4,10000)`. `Y` nằm trong `jasperRidge2_R198.mat`; `M`, `A` nằm trong `Jasper_GT.mat`. Thứ tự lớp GT là **tree, water, dirt, road**, khớp chính xác với cả hai FIG, không cần hoán vị lớp.

`M` trùng từng giá trị với phổ `YData` của FIG Materials. `A[k,:].reshape(100,100,order='F')` trùng từng giá trị với `CData` tương ứng của FIG Abundance. Điều này xác minh thứ tự pixel của **A so với FIG**, chưa xác minh độc lập thứ tự pixel của **Y so với A**.

Scale chung giữa `Y` và `M` **chưa được chứng minh**: metadata `maxValue=5000` là bằng chứng trực tiếp duy nhất cho một giá trị scale ứng viên của Y, nhưng không có công thức chuẩn hóa đi kèm. FIG xác nhận scale hiện có của M, không chứa cầu nối từ số nguyên Y sang M. Không dùng fit `M@A`, tối ưu hệ số, hay nhãn GT để chọn mẫu số.

## Phạm vi và khả năng tái lập

Chỉ đọc bốn file MAT/FIG liệt kê dưới đây bằng `scipy.io.whosmat` và `scipy.io.loadmat`. FIG được duyệt dưới dạng struct, không mở MATLAB, không thực thi callback/listener. Không đọc PDF, không chạy EDA, không chạy hoặc sửa pipeline, không cài thư viện. Chỉ tạo hai file được cho phép: báo cáo này và `scripts/audit_jasper_source.py`, bằng `apply_patch`.

Môi trường thực chạy: Python 3.12.10, NumPy 2.2.5, SciPy 1.15.2. Chạy từ workspace:

```powershell
python scripts/audit_jasper_source.py
python scripts/audit_jasper_source.py --inspect
```

Script chỉ in JSON ra stdout; không ghi file dữ liệu hay artifact phụ. Chế độ `--inspect` liệt kê thống kê mọi trường số lồng trong MAT/FIG; chế độ mặc định giữ trường đồ họa liên quan và toàn bộ kết quả đối chiếu. Phép ghép duyệt đủ 24 hoán vị lớp với khoảng cách max-absolute-error; mọi phép trừ giữa phổ/abundance đều ở float64.

Lưu ý dtype: `whosmat` báo MATLAB class `double` cho các trường số của R198, trong khi `loadmat` mặc định (`mat_dtype=False`) trả về `Y:uint16`, `SlectBands:uint8` và các scalar integer ghi trong bảng. Hai thông tin mô tả các lớp khác nhau của file MAT, không phải lỗi nội dung. Bảng ghi dtype NumPy thực đọc; giữ nguyên shape bằng đọc không squeeze. `simplify_cells=True` chỉ dùng để duyệt cấu trúc FIG và nhãn khi đối chiếu, không dùng suy ra dtype scalar nguồn.

## Danh mục và SHA256 toàn bộ byte nguồn

| File trong `data/` | Số byte | SHA256 |
|---|---:|---|
| `Jasper_GT.mat` | 178271 | `92f5697b43705802b904fd13ba99b6ce65a3d203682864abc3fbec922beec374` |
| `jasperRidge2_R198.mat` | 2987224 | `0e4118a6452f6044978a8ca3762fb0f791115467904936d463c4e111e56e682e` |
| `end4_Abundance.fig` | 179847 | `1d0ce4237d03ac97cd7f64debde8e79d2a526f050357a8bdbc0efdee8be2b857` |
| `end4_Materials.fig` | 13627 | `7870d2be51e88da548665ff5093a0216f2c894d5630464a0219c395b829a13ad` |

MAT header tự khai: GT `PCWIN64`, `Wed Oct 22 15:31:33 2014`; R198 `PCWIN64`, `Wed Oct 22 15:26:40 2014`; Abundance FIG `GLNXA64`, `Mon Sep 24 20:14:53 2012`; Materials FIG `GLNXA64`, `Mon Sep 24 20:14:52 2012`. Đây là chuỗi trong header, không phải bằng chứng độc lập về tác giả hay ngày thu nhận ảnh.

## Keys, shapes và thống kê số

`Jasper_GT.mat` có đúng các key dữ liệu `cood`, `M`, `A`; không có `Y`. `cood` là cell `(4,1)`, nội dung nguyên văn `1-tree`, `2-water`, `3-dirt`, `4-road`; đây là nhãn, không phải tọa độ pixel.

`jasperRidge2_R198.mat` có đúng `Region`, `SlectBands`, `nRow`, `nCol`, `nBand`, `Y`, `maxValue`; không có `M`, `A` ở top-level hoặc trong `Region`. Các key `__header__`, `__version__`, `__globals__` do SciPy cung cấp là metadata đọc file.

| File / trường | Shape | Dtype NumPy | Min | Max | NaN | Inf | Số phần tử bằng 0 |
|---|---|---|---:|---:|---:|---:|---:|
| R198 / Y | (198,10000) | uint16 | 0 | 5437 | 0 | 0 | 418 |
| GT / M | (198,4) | float64 | 0 | 0.6290566037735849 | 0 | 0 | 3 |
| GT / A | (4,10000) | float64 | 0 | 1.0000000000000002 | 0 | 0 | 17197 |
| R198 / SlectBands | (198,1) | uint8 | 4 | 219 | 0 | 0 | 0 |
| R198 / nRow | (1,1) | uint8 | 100 | 100 | 0 | 0 | 0 |
| R198 / nCol | (1,1) | uint8 | 100 | 100 | 0 | 0 | 0 |
| R198 / nBand | (1,1) | uint8 | 224 | 224 | 0 | 0 | 0 |
| R198 / maxValue | (1,1) | uint16 | 5000 | 5000 | 0 | 0 | 0 |
| R198 / Region.xStart | (1,1) | uint16 | 269 | 269 | 0 | 0 | 0 |
| R198 / Region.xEnd | (1,1) | uint16 | 368 | 368 | 0 | 0 | 0 |
| R198 / Region.yStart | (1,1) | uint8 | 105 | 105 | 0 | 0 | 0 |
| R198 / Region.yEnd | (1,1) | uint8 | 204 | 204 | 0 | 0 | 0 |

`Region` là struct `(1,1)` với đúng bốn trường trên. Chênh lệch hai đầu cộng một đều bằng 100, phù hợp kích thước crop; chưa có mã nguồn để xác nhận x/y ứng với row/column của ảnh gốc như thế nào.

Không có band hoặc pixel Y toàn 0; không có cột A toàn 0. Chưa có mask hay quy tắc missing-data giải thích 418 số 0 trong Y, vì vậy không tự coi chúng là nodata hoặc sửa chúng.

### Band metadata

`SlectBands` gồm 198 chỉ số duy nhất, tăng nghiêm ngặt: `4–107`, `113–153`, `167–219` (các khoảng tính cả hai đầu). Nếu diễn giải trên hệ chỉ số MATLAB 1–224, các band không nằm trong danh sách là `1–3`, `108–112`, `154–166`, `220–224`, tổng cộng 26.

`nBand=224` phù hợp metadata số band trước lựa chọn; số band hiện có phải lấy từ `Y.shape[0]=198`. Không áp dụng `SlectBands` lần nữa lên Y đã có 198 hàng. File không cung cấp wavelength hoặc giải thích vật lý cho từng khoảng band bị loại; việc M có 198 hàng chỉ xác nhận số chiều, chưa độc lập xác nhận ánh xạ wavelength của M với R198.

### Tổng abundance

Tính `A.sum(axis=0)` cho 10000 pixel: min `0.9999999999999808`, max `1.0000000000000002`, sai số tuyệt đối lớn nhất so với 1 là `1.9206858326015208e-14`. Có 1971 tổng không bằng 1 theo so sánh float chính xác, nhưng **0** tổng lệch quá `1e-12`. Không có abundance âm; có đúng **1** phần tử lớn hơn 1, với mức vượt `2.220446049250313e-16`, phù hợp sai số số thực. Chưa có lý do từ audit để clip hoặc chuẩn hóa lại A.

## Bằng chứng từ FIG và phép ghép lớp

Cả hai FIG có top-level `hgS_070000`, struct `(1,1)`. Duyệt các `children` theo thứ tự lưu cho ra bốn axes; mỗi axes chứa một đối tượng dữ liệu và text nhãn. Các đường dẫn dưới đây dùng chỉ số Python từ 0 sau `simplify_cells=True`:

- Abundance: `hgS_070000.children[k].children[0].properties.CData`, loại `image`.
- Materials: `hgS_070000.children[k].children[0].properties.YData`, loại `graph2d.lineseries`.
- Nhãn mỗi axes: `children[1].properties.String`, lần lượt `1-tree`, `2-water`, `3-dirt`, `4-road`.

| k / nhãn nguyên văn | Materials YData min | Materials YData max | Zeros phổ | Abundance CData min | Abundance CData max | Zeros bản đồ |
|---|---:|---:|---:|---:|---:|---:|
| 0 / 1-tree | 0 | 0.5558490566037736 | 1 | 0 | 1 | 2500 |
| 1 / 2-water | 0 | 0.1432145352900069 | 1 | 0 | 1 | 5150 |
| 2 / 3-dirt | 0 | 0.6290566037735849 | 1 | 0 | 1 | 3800 |
| 3 / 4-road | 0.04396226415094339 | 0.6050943396226415 | 0 | 0 | 1.0000000000000002 | 5747 |

Mỗi `YData` Materials có shape lưu `(1,198)`, dtype float64; `XData` `(1,198)`, uint8, bằng chính xác dãy `1..198`, không phải wavelength. Mỗi `CData` Abundance có shape `(100,100)`, dtype float64; `XData` và `YData` ảnh đều `(1,2)`, uint8, giá trị `[1,100]`; `CLim=[0,1]`, `YDir='reverse'`, `CDataMapping='scaled'`. Các vector tọa độ đều không có NaN/Inf/zero; `CLim` có một số 0. Toàn bộ phổ và CData trong bảng có NaN=Inf=0.

FIG còn có giá trị số phức lớn trong metadata đồ họa `SubplotListeners`/`SubplotDeleteListener` (5 trường trong mỗi FIG, phần thực -1, phần ảo khoảng `1.63456784214e301`). Đây là dữ liệu serialized listener, không đưa vào min/max phổ hoặc abundance và không thực thi. Script tách thống kê phần thực/phần ảo; không phát hiện NaN/Inf trong các trường số đã duyệt.

| Đối chiếu đủ 24 hoán vị lớp | Hoán vị duy nhất cho sai số 0 | Max abs error tốt nhất | Max abs error hoán vị tốt thứ hai |
|---|---|---:|---:|
| M[:,k] với Materials YData | [0,1,2,3] | 0 | 0.20886792452830188 |
| A[k,:] với CData.ravel(order='F') | [0,1,2,3] | 0 | 1 |
| A[k,:] với CData.ravel(order='C') | Không có | 1 | 1 |

Vì R198 chỉ có Y, không có bộ M/A thứ hai hoặc nhãn lớp, **không thể thực hiện phép ghép hoán vị endmember GT ↔ R198 trực tiếp**. Phép ghép thực sự kiểm được ở đây là GT ↔ FIG; không được trình bày nó như chứng minh tương ứng lớp/pixel của R198. Không dùng giải unmixing từ Y để tạo nhãn phục vụ phép ghép.

### Pixel order: điều đã chứng minh và giới hạn

Với cả bốn lớp, đẳng thức sau đúng từng phần tử, không dùng tolerance:

```python
A[k, :] == CData_k.ravel(order='F')
CData_k == A[k, :].reshape((100, 100), order='F')
```

Theo chỉ số từ 0, pixel `p = row + 100*col`; `row = p % 100`, `col = p // 100`. `YDir='reverse'` đặt hàng đầu ở phía trên của ảnh hiển thị trong FIG, không phải bằng chứng về hướng địa lý. Trải C trực tiếp không khớp, kể cả cho phép đổi thứ tự bốn lớp. Audit đối chiếu hai quy tắc flatten C/F; không tuyên bố đã loại trừ mọi hoán vị pixel hoặc mọi phép xoay/lật có thể có.

Không có ảnh band/RGB của Y trong hai FIG, không có chỉ số pixel liên kết Y với A, và không có mã tạo file trong bốn nguồn này. Vì vậy đề xuất dùng F cho cube Y chỉ là quy ước dự kiến nếu nguồn gốc chung được xác minh thêm, chưa phải kết quả chứng minh từ FIG. Giữ Y dưới dạng ma trận hai chiều giúp tránh quyết định reshape chưa có bằng chứng.

## Scale: bằng chứng và mức tin cậy

| Nhận định | Bằng chứng trực tiếp | Mức tin cậy / giới hạn |
|---|---|---|
| Scale M hiện tại đúng với FIG Materials | 792/792 giá trị M bằng YData sau ghép identity | Cao về tính đồng nhất số; không xác nhận đơn vị vật lý |
| File R198 ghi maxValue=5000 | Scalar đọc nguyên văn từ MAT | Cao về metadata; chưa xác nhận ý nghĩa là divisor |
| 5000 không phải cực đại thực của Y | Y.max()=5437, 18 phần tử >5000 | Cao |
| Y/5000 có miền [0,1.0874] | Phép chia float64 bằng metadata, chỉ tính trong RAM | Cao về số học; chưa xác nhận đây là reflectance cùng scale với M |
| Y và M có cùng quy ước hiệu chuẩn sau /5000 | Không có công thức, đơn vị, mã tạo file hoặc phổ raw liên kết | Chưa xác minh |
| Chọn divisor khác, ví dụ 5300, từ dạng số hoặc độ khớp GT | Không có trường metadata hỗ trợ trong nguồn đã đọc | Không đủ bằng chứng để khuyến nghị |

FIG chỉ cho phổ M đã ở miền xấp xỉ 0–0.63 và trục band `1..198`; nó không ghi divisor hay đơn vị reflectance. `CDataMapping='scaled'` và `CLim=[0,1]` điều khiển màu bản đồ abundance, không phải phép hiệu chuẩn Y. Không thể lấy tính tương tự biên độ M và Y/5000 làm chứng minh scale.

Đề xuất giữ `Y_raw` cùng metadata gốc; nếu bước sau cần khảo sát phép chia 5000, phải đặt tên là giả thuyết dựa trên metadata và xác minh bằng nguồn sơ cấp/mã xử lý. Không chia M thêm lần nữa chỉ vì Y là integer. Không clip 18 giá trị Y/5000 >1, không thay `maxValue` thành 5437, và không chọn scale để giảm residual với GT trong audit này.

## Nguồn GitHub: chỉ điều Git cục bộ chứng minh

`git remote -v` trả origin fetch/push là [LloydPhuc/Quantum-Nonlinear-Unmixing](https://github.com/LloydPhuc/Quantum-Nonlinear-Unmixing). Đây là URL remote **đã cấu hình trong repo cục bộ**; audit không kiểm tra mạng và không tuyên bố đã xác minh khả dụng của URL hoặc tác giả dữ liệu gốc.

HEAD lúc kiểm tra: `504edcfa27d1dbaeb0c2bcee1b43ba66b141dcf0`. `git log --all -- data` và `git log --follow` cho từng file đều dẫn tới commit thêm bốn file `f4defc167142352bb9daab8e83e4ccf62b3710d5`, thông điệp `Initial commit: QCNN Unmixing Pipeline with IESA architecture`; AuthorDate `2026-09-11 23:07:57 +0700`, CommitDate `2026-09-11 23:15:11 +0700`. Không có lịch sử data cũ hơn trong lịch sử cục bộ đã kiểm tra.

Git blob IDs ở HEAD (khác với SHA256 toàn file ở trên):

| File | Git blob ID |
|---|---|
| Jasper_GT.mat | `2b6792fd0049c8327a2547acc74b0aabdacc4c89` |
| jasperRidge2_R198.mat | `7ca30c40f8a2e4d55477f5ce5722576b0150e8d4` |
| end4_Abundance.fig | `b11a15897901a19dd34f469cba2f8cf1bc57a376` |
| end4_Materials.fig | `a5dfec71371989163971bc2ecc7b22bf0299aca8` |

`git diff --name-only HEAD -- data` không có output. SHA256 đọc lại trùng lần đọc đầu. Tìm kiếm văn bản phạm vi `README.md`, `data`, `scripts`, `src`, `configs` không tìm thấy công thức scale hoặc dẫn nguồn Jasper cụ thể; chỉ có tên dataset trong config legacy. Remote và commit nhập dữ liệu chưa chứng minh nơi tải ban đầu, giấy phép, nhà phát hành benchmark, hoặc liên hệ giữa các lần lưu 2012/2014. Agent chính đang xác minh nguồn sơ cấp; audit này không suy đoán một GitHub upstream khác từ tên file.

## Đề xuất canonical và những điểm còn mở

| Thành phần canonical đề xuất | Nguồn | Quy tắc được dữ liệu hỗ trợ |
|---|---|---|
| Y_raw: (198,10000), uint16 khi load mặc định | R198 / Y | Giữ thứ tự hàng/cột và giá trị hiện có; chưa chốt scale hay reshape không gian |
| M: (198,4), float64 | GT / M | Giữ nguyên; cột tree, water, dirt, road; trùng FIG |
| A: (4,10000), float64 | GT / A | Giữ nguyên; hàng tree, water, dirt, road; tổng mỗi cột gần 1 |
| Bản đồ GT: (4,100,100) | Các hàng A | Reshape từng hàng `order='F'`; đã chứng minh với FIG |
| selected_bands: 198 chỉ số | R198 / SlectBands | Lưu cả quy ước 1-based của nguồn; nếu tạo index Python thì chuyển kiểu rộng trước thao tác |

Các unknown cần nguồn bổ sung: divisor/đơn vị chung Y–M; thứ tự cột Y tương ứng với A; ánh xạ band/wavelength M–Y; ý nghĩa vật lý của zeros; quy trình tạo GT, độ chính xác nhãn, giấy phép và upstream trước commit nhập dữ liệu. Audit hoàn tất kiểm tra dữ liệu cục bộ, nhưng chưa đủ cơ sở tuyên bố bộ ba đã được đồng bộ hoàn toàn cho đánh giá mô hình.

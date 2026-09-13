# Đối chiếu PDF và quyết định cho bản Jasper Ridge local

Ngày kiểm chứng: 2026-09-13. Tài liệu đầu vào: `directive/Tiền Xử Lý Jasper Ridge.pdf`, 13 trang. Đã đọc text tất cả trang và xem bản render trang 6-9 vì nhiều công thức/số biến mất khi trích text. Đây là bản ghi quyết định của orchestrator; số đo dữ liệu trong source audit và EDA report.

| Đề xuất/khẳng định trong PDF | Bằng chứng đối chiếu | Quyết định local |
|---|---|---|
| Bỏ 26 band từ 224, crop 100x100 | MAT Y=(198,10000), SlectBands/Region; Zhu §V-B | Hai bước đã làm; chỉ validate, không làm lại |
| Chia /6000 hoặc /10000 | MAT maxValue=5000; Borsoi demo /6000 nhưng dùng M0_jasperRidge.mat khác | Scale Y theo 5000, giữ M fractional; mức xác nhận bằng metadata và consistency audit, không tuyên bố đơn vị reflectance vật lý được hiệu chuẩn |
| Thứ tự Road, Soil, Water, Tree | `cood` local: tree, water, dirt, road | Giữ local order và kiểm đối chiếu FIG; không copy thứ tự từ một bản phân phối khác |
| Jasper endmembers đối sánh thư viện USGS | Zhu §IV-A phân biệt chọn phổ từ ảnh Jasper/Samson và từ thư viện cho Cuprite | Không mặc định M của Jasper được lấy từ USGS |
| Ground truth có ANC/ASC, sinh bằng constrained LS | Zhu §IV-B mô tả abundance labeling tối ưu, có thể dùng thêm priors; chưa có log tạo file GT cụ thể | Gọi A_ref là reference estimates; không khẳng định solver lịch sử chính xác và không gọi đây là tỷ lệ đo thực địa |
| Dữ liệu int16 | Y local là uint16 | Dùng dtype thực; cast float64 trước tính toán để tránh unsigned underflow |
| Chuẩn hóa global loại bỏ bóng râm | Phép chia một scalar giữ tỷ lệ độ sáng giữa pixels | Không có bảo đảm đó; giữ các biến thiên để phân tích |
| L2 từng pixel hoặc PCA giúp xử lý | Là các biến đổi phục vụ mục tiêu khác nhau | Không dùng trên clean forward-model input; PCA fit train chỉ phục vụ EDA |
| MNF/PCA tách hoàn toàn noise, VD gần 4 | Phụ thuộc model noise/dữ liệu; chưa chạy HySime/MNF | Chỉ báo PCA variance thực đo, không suy ra số endmember hay noise-free signals |
| Color composite 145/99/19 là NIR/red/green | MAT không cung cấp central wavelengths cho selected bands | Composite minh họa theo retained/original IDs, không gán vùng bước sóng chưa xác minh |

Nguồn đã đọc:

- [Zhu, Hyperspectral Unmixing: Ground Truth Labeling, Datasets, Benchmark Performances and Survey (2017), §IV và §V-B](https://ar5iv.labs.arxiv.org/html/1708.05125). Phương pháp labeling dùng phổ chọn từ ảnh hoặc thư viện tùy dataset, abundance do tối ưu; xác minh hình ảnh và tính nhất quán, không tương đương đo abundance độc lập.
- [Trang phân phối dữ liệu dẫn về Feiyun Zhu](https://lesun.weebly.com/hyperspectral-data-set.html). Xác nhận ROI và band list. Trang ghi research/education only; không coi đây là bằng chứng đầy đủ về license/provenance của mirror GitHub local.
- [Borsoi et al., demo_jasperRidge.m](https://github.com/ricardoborsoi/MultiscaleKernelSURelease/blob/master/demo_jasperRidge.m). Mã load Y, reshape MATLAB và chia 6000; M được load riêng. Đây là ví dụ một convention, không phải căn cứ để chia Y local khác thang với M local.

Git remote của workspace không tự chứng minh repository nguồn nơi người dùng tải MAT/FIG. Nếu không có URL/commit gốc được ghi trong lịch sử thì manifest ghi unknown; hashes định danh chính xác bản đang phân tích. Không tải thay thế các raw files.

Với QSiMix, việc học residual trên A_ref đánh giá khả năng khớp một benchmark reference. Sai khác `Y-M@A_ref` có thể gồm noise, variability, scale mismatch và label mismatch. Không diễn giải toàn bộ residual là scattering phi tuyến, và không dùng reconstruction giảm để suy ra abundance vật lý chính xác hơn.

# OCR Sổ đỏ / Sổ hồng

## 1. Giới thiệu

Đề tài: **Phân tích, phát hiện dấu hiệu cắt ghép/chỉnh sửa và trích xuất dữ liệu tự động từ Giấy chứng nhận quyền sử dụng đất (Sổ đỏ / Sổ hồng).**

Hệ thống gồm các phần chính:

- OCR và trích xuất các trường thông tin ở Trang 2.
- Kiểm tra dấu hiệu chỉnh sửa chữ/số ở Trang 2.
- Phân tích sơ đồ, copy-move và bất thường cục bộ ở Trang 3.
- Kiểm tra vùng con dấu bằng HSV + LBP.
- Giao diện Streamlit và xuất kết quả JSON/Excel.

> Lưu ý: hệ thống chỉ hỗ trợ **sàng lọc dấu hiệu bất thường/chỉnh sửa**, không phải công cụ xác thực pháp lý hay kết luận gian lận.

---

## 2. Môi trường đã kiểm thử

Bản release đã được kiểm thử với:

- Windows 10 64-bit
- Python **3.10.11**
- pip **26.2.1**
- PaddlePaddle **3.3.1**
- PaddleOCR **3.7.0**
- PaddleX **3.7.2**
- OpenCV contrib **4.10.0.84**
- NumPy **1.26.4**
- scikit-learn **1.7.2**
- scikit-image **0.25.2**
- Streamlit **1.65.0**

Không khuyến nghị dùng Python 3.14 để tạo môi trường cho project.

---

## 3. Cấu trúc chính

```text
OCR_SoDo/
├── dataset/
├── docs/
├── src/
│   ├── tuan1_dataset_tool_auto.py
│   ├── tuan1_dataset_tool_semi_auto.py
│   ├── tuan2_preprocess.py
│   ├── tuan3_baseline.py
│   ├── tuan4_pipeline.py
│   ├── tuan5_evaluate.py
│   └── tuan7_app.py
├── Tesseract/
├── check_system.py
├── run_with_heartbeat.py
├── requirements.txt
├── requirements-release.txt
├── setup_windows.bat
└── run_project.bat
```

---

## 4. Cài đặt

### Bước 1 - Chuẩn bị Python

Cần Python **3.10.x 64-bit**.

Nếu dùng Conda, có thể activate một environment Python 3.10 trước khi chạy setup.

### Bước 2 - Chạy setup

Tại thư mục gốc project, chạy:

```bat
setup_windows.bat
```

Script sẽ:

1. Tìm Python 3.10.x (bản tham chiếu là 3.10.11 64-bit).
2. Nếu chưa có Python 3.10, hỏi `Y/N`; chọn `Y` để mở trang Python 3.10.11 chính thức và tự tải **Windows installer (64-bit)**.
3. Tạo `.venv`.
4. Cài pip 26.2.1 và dependency đã khóa.
5. Chạy `pip check`.
6. Kiểm tra OpenCV, PaddlePaddle, PaddleOCR và Tesseract.
7. Có thể chạy `check_system.py` ngay ở cuối setup.

Nếu setup hoàn tất, màn hình sẽ hiện:

```text
SETUP HOAN TAT
```

---

## 5. Kiểm tra hệ thống

Sau khi setup, chạy:

```bat
.venv\Scripts\python.exe check_system.py
```

Kết quả tốt nhất:

```text
SYSTEM READY
Fatal errors : 0
```

`check_system.py` kiểm tra:

- Python 3.10
- `.venv`
- đủ 7 file source
- compile source
- dependency chính
- OpenCV SIFT/ORB
- PaddlePaddle/PaddleOCR
- Tesseract
- dataset và dữ liệu preprocess

---

## 6. Chạy project

Cách đơn giản nhất:

```bat
run_project.bat
```

Menu chạy các script qua `run_with_heartbeat.py`. Nếu một công đoạn im lặng khoảng 12 giây, terminal sẽ tự báo rằng chương trình vẫn đang xử lý. Điều này tránh hiểu nhầm chương trình bị lag/treo và không làm thay đổi thuật toán.

Menu gồm:

```text
1. Tuần 1 - Dataset / che PII / synthetic tamper
2. Tuần 2 - Preprocess / tách ROI
3. Tuần 3 - HSV / FFT / Tesseract baseline
4. Tuần 4 - Proposed pipeline
5. Tuần 5 - Đánh giá thực nghiệm
7. Tuần 7 - Streamlit demo
0. Thoát
```

### Thứ tự khuyến nghị

```text
Tuần 1
→ Tuần 2
→ Tuần 3
→ Tuần 4
→ Tuần 5
→ Tuần 7
```

Tuần 3/4/5/7 sử dụng dữ liệu preprocess, vì vậy nên chạy Tuần 2 trước nếu chưa có `dataset/processed_*`.

---

## 7. Nội dung từng tuần

### Tuần 1 - Dataset

- Ẩn thông tin cá nhân cần thiết.
- Sinh synthetic tamper cho text, diagram và seal.
- Có hai chế độ:
  - AUTO
  - SEMI-AUTO
- Khi chọn AUTO từ `run_project.bat`, chương trình tự đếm số ảnh trong `dataset/raw` và cho phép nhập số ảnh muốn chạy. Nhấn Enter để chạy toàn bộ.
- AUTO hiển thị từng bước OCR/PII/tamper, phần trăm tiến độ, thời gian mỗi ảnh và ETA ước tính.
- Chạy subset chỉ phù hợp smoke-test; metric Tuần 5 có thể khác báo cáo full dataset.

Mốc dữ liệu đã kiểm thử:

```text
raw            : 78
real           : 78
tamper_text    : 31
tamper_diagram : 78
tamper_seal    : 59
tampered_total : 168
```

### Tuần 2 - Preprocess

- Perspective correction / deskew.
- Resize.
- CLAHE.
- Tách ROI text / diagram / seal.

Mốc output đã kiểm thử:

```text
Real mỗi ROI  : 78
Text tamper   : 31
Diagram tamper: 78
Seal tamper   : 59
```

### Tuần 3 - Baseline

- HSV tách vùng đỏ.
- FFT so sánh đặc trưng tần số.
- Tesseract baseline.

### Tuần 4 - Proposed Pipeline

- Geometry: Skeletonization + HoughLinesP.
- OCR: Morphology Dilate + PaddleOCR.
- Seal: HSV + LBP.

### Tuần 5 - Đánh giá

Tamper benchmark:

```text
Accuracy          : 0.6972
Precision         : 0.4773
Recall            : 0.6774
F1-score          : 0.5600
AUC-ROC           : 0.7275
```

OCR controlled synthetic benchmark:

```text
CER        : 0.00%
WER        : 0.00%
Exact Match: 100.00%
```

Geometry controlled synthetic benchmark:

```text
Detection rate: 100.00%
MAE           : 39.367 px
RMSE          : 70.670 px
MAPE          : 13.56%
```

Các kết quả OCR/Geometry trên đây là **controlled synthetic benchmark**, không phải ground truth đại diện cho toàn bộ ảnh Sổ đỏ thực tế.

### Tuần 7 - Streamlit Demo

Khi mở từ `run_project.bat`, Streamlit chạy ở chế độ headless nên không hỏi email lần đầu. Menu tự chọn một port trống, chờ server phản hồi rồi mở trình duyệt; URL luôn được in ra để có thể mở thủ công.

Trang 2:

- OCR nhiều lượt.
- Trích xuất trường dữ liệu.
- Text tamper screening.

Trang 3:

- Local anomaly.
- Copy-move.
- Seam.
- Seal screening bằng HSV + LBP.

Xuất:

- JSON
- Excel

---

## 8. Lưu ý về dataset

Dataset thực nghiệm chủ yếu là ảnh **Trang 2 và Trang 3** do giới hạn nguồn dữ liệu công khai và yêu cầu riêng tư.

Không phải mọi ảnh trong `anonymized_real` đều bắt buộc có vùng blur. Nếu ảnh không có thông tin cá nhân cần ẩn thì ảnh có thể giữ nguyên.

Các trường như:

- Số thửa
- Tờ bản đồ
- Diện tích
- Địa chỉ thửa đất
- Mục đích sử dụng
- Thời hạn sử dụng
- Nguồn gốc sử dụng

được giữ để phục vụ OCR và phân tích.

---

## 9. Tesseract

Project có thể dùng bản local:

```text
Tesseract\tesseract.exe
```

Tesseract chỉ phục vụ baseline Tuần 3.

OCR chính ở Tuần 4/5/7 dùng PaddleOCR.

---

## 10. Lỗi Windows Application Control / Defender

Nếu gặp lỗi dạng:

```text
ImportError: DLL load failed ...
An Application Control policy has blocked this file.
```

đặc biệt với:

```text
libpaddle.pyd
```

thì đây thường là chính sách bảo mật Windows chặn native DLL/.pyd.

Đây không phải lỗi cú pháp Python của project.

Không nên tắt toàn bộ Windows Defender.

Có thể:

- thử trên máy cá nhân khác không bị policy hạn chế;
- nhờ quản trị viên cho phép binary cần thiết;
- chạy `check_system.py` để xác định package bị chặn.

---

## 11. Ghi chú về kết quả phát hiện chỉnh sửa

Các trạng thái như:

- Chưa phát hiện chỉnh sửa
- Cần kiểm tra
- Nghi vấn chỉnh sửa

là kết quả **screening**.

Tamper score không phải xác suất gian lận và không thay thế giám định pháp lý.

Synthetic tamper được sử dụng phục vụ thí nghiệm học thuật; kết quả cần được diễn giải trong phạm vi benchmark có kiểm soát.

---

## 12. Quy trình chạy nhanh

```text
1. Giải nén project
2. Chạy setup_windows.bat
3. Chạy:
   .venv\Scripts\python.exe check_system.py
4. Khi thấy SYSTEM READY:
   chạy run_project.bat
```

Nếu chỉ muốn mở giao diện:

```bat
.venv\Scripts\python.exe -m streamlit run src\tuan7_app.py
```


## 13. Tốc độ và chế độ test nhanh

- Tuần 1 là phần tốn thời gian nhất do PaddleOCR chạy trên từng ảnh. Cách an toàn để test nhanh là chọn ít ảnh trực tiếp từ menu Tuần 1 AUTO.
- PaddleX model-source connectivity check được tắt khi chạy từ menu để giảm chờ không cần thiết; model OCR đã tải sẽ được dùng lại từ cache.
- Tuần 4 và Tuần 5 giữ nguyên cấu hình benchmark để không làm thay đổi kết quả đã báo cáo. Heartbeat sẽ báo định kỳ nếu một bước xử lý lâu.
- Không song song hóa PaddleOCR CPU và không đổi sang model nhỏ hơn trong bản release vì có thể làm thay đổi kết quả hoặc độ ổn định.

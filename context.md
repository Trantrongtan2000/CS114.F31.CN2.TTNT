# 📋 CONTEXT & HƯỚNG DẪN DỰ ÁN CHO AI AGENTS — CS114

> **Môn học:** CS114 — Nhập Môn Máy Học (Introduction to Machine Learning)  
> **Nhóm thực hiện:** Nhóm đồ án UIT-VSMEC (4 thành viên):  
> 1. **Lê Quang Thi** — MSSV: `25210337` (Nhóm trưởng)  
> 2. **Trần Trọng Tấn** — MSSV: `25210334`  
> 3. **Nguyễn Quang Lâm** — MSSV: `25210289`  
> 4. **Võ Cẩm Thu** — MSSV: `25210342`  
> **Giảng viên phụ trách:** Khoa Khoa học Máy tính — Trường Đại học Công nghệ Thông tin (ĐHQG-HCM)

---

## 🎯 1. NỘI DUNG ĐỒ ÁN
* **Tên đề tài chính thức:** **Nhận diện Cảm xúc trên Mạng xã hội Tiếng Việt (Vietnamese Social Media Emotion Recognition — UIT-VSMEC)**
* **Mục tiêu học thuật:**
  * Xây dựng pipeline học máy hoàn chỉnh giải quyết bài toán phân loại cảm xúc đa lớp (7 nhãn) trên ngôn ngữ mạng xã hội tiếng Việt.
  * Thiết kế ma trận thực nghiệm 2 chiều: **3 Biểu diễn đặc trưng (Word TF-IDF, Char n-grams TF-IDF, Hybrid)** $\times$ **4 Họ mô hình học máy (Multinomial Naive Bayes, Logistic Regression, Linear SVM, Random Forest)** = **12 cấu hình đối sánh**.
  * So sánh chiến lược phân loại đa lớp (Buổi 04): **One-vs-Rest (OvR - 7 models)** vs **One-vs-One (OvO - 21 models)**.
  * Phân tích chuyên sâu hiện tượng **Quá khớp (Overfitting / Underfitting)** và **Đánh đổi Bias - Variance** (Buổi 05) qua đồ thị Learning Curves và Regularization Path.
  * Bàn giao: Mã nguồn Git chuẩn mực, Bảng tổng hợp số liệu `metrics_summary.csv`, 5 biểu đồ trực quan hóa báo cáo, Báo cáo đồ án Word/PDF < 15 trang.

---

## 📊 2. BỘ DỮ LIỆU (DATASET) & ĐẶC TÍNH
* **Bộ dữ liệu sử dụng chính:** **`UIT-VSMEC` (Vietnamese Social Media Emotion Corpus)**:
  * Do nhóm nghiên cứu NLP của Trường ĐH Công nghệ Thông tin (UIT) công bố.
  * Quy mô: 6.927 câu bình luận tiếng Việt phân bổ theo chuẩn benchmark:
    * `train.csv`: 5.548 câu (80,1%)
    * `val.csv`: 686 câu (9,9%)
    * `test.csv`: 693 câu (10,0%)
  * 7 nhãn cảm xúc: `Enjoyment` (28,1%), `Disgust` (19,3%), `Other` (18,4%), `Sadness` (17,1%), `Anger` (7,0%), `Fear` (5,7%), `Surprise` (4,4%).
  * Hiện tượng mất cân bằng lớp (Class Imbalance) rõ rệt $\rightarrow$ bắt buộc sử dụng **Macro-F1** và **Confusion Matrix Heatmap**.

---

## 👨‍🏫 3. TÍNH CÁCH & TIÊU CHÍ ĐÁNH GIÁ CỦA GIẢNG VIÊN (RÚT RA TỪ 5 VIDEO BÀI GIẢNG)
* **Phong cách giảng dạy:** Hướng đến bản chất thuật toán và quy trình thực nghiệm khách quan; yêu cầu tính toán đo đạc chuẩn mực và hiểu sâu sắc hiện tượng Bias-Variance.
* **Gu đánh giá & Tiêu chí chấm điểm cốt lõi:**
  1. **Quy trình chuẩn chỉ (Full Pipeline):** Bắt buộc có đầy đủ các bước: EDA trực quan $\rightarrow$ Tiền xử lý (xử lý teencode, emoji, phân đoạn từ) $\rightarrow$ Trích xuất đặc trưng $\rightarrow$ Huấn luyện $\rightarrow$ Đánh giá.
  2. **So sánh ít nhất 3 mô hình cơ bản:** Phải so sánh đối sánh giữa các họ mô hình khác nhau (xác suất, tuyến tính, biên cực đại, ensemble).
  3. **Không được lừa dối bằng Accuracy:** Vì dữ liệu mất cân bằng lớp (Surprise 4,4% vs Enjoyment 28,1%), báo cáo phải chứng minh bằng **Macro-Precision, Macro-Recall, Macro-F1** và **Confusion Matrix**.
  4. **Phân tích Overfitting / Underfitting (Điểm phân loại cao):** Phải có biểu đồ **Learning Curve** (thay đổi kích thước tập huấn luyện từ 20% đến 100%) và phân tích khoảng cách sai biệt giữa Train và Validation.
  5. **Báo cáo gọn gàng, súc tích:** Báo cáo Word dưới 15 trang, hình vẽ rõ nét, bảng biểu đầy đủ.

---

## 📂 4. DANH MỤC FILE & VAI TRÒ MÃ NGUỒN
* [`data/vsmec/`](file:///home/tan/04_Studies_Knowledge/UIT_Studies/Các môn học kỳ 3/CS114/data/vsmec/): Chứa các file dữ liệu `train.csv`, `val.csv`, `test.csv` và bản cleaned cache.
* [`src/preprocessor.py`](file:///home/tan/04_Studies_Knowledge/UIT_Studies/Các môn học kỳ 3/CS114/src/preprocessor.py): Làm sạch teencode, emoji, phân đoạn từ pyvi.
* [`src/feature_builder.py`](file:///home/tan/04_Studies_Knowledge/UIT_Studies/Các môn học kỳ 3/CS114/src/feature_builder.py): Xây dựng 3 dạng TF-IDF (Word, Char, Hybrid).
* [`src/models.py`](file:///home/tan/04_Studies_Knowledge/UIT_Studies/Các môn học kỳ 3/CS114/src/models.py): Khởi tạo 4 mô hình ML và bộ bọc OvR / OvO.
* [`src/evaluate.py`](file:///home/tan/04_Studies_Knowledge/UIT_Studies/Các môn học kỳ 3/CS114/src/evaluate.py): Tính toán các độ đo và định lượng Train-Val Gap.
* [`src/plot_utils.py`](file:///home/tan/04_Studies_Knowledge/UIT_Studies/Các môn học kỳ 3/CS114/src/plot_utils.py): Xuất 5 biểu đồ phục vụ báo cáo.
* [`run_experiments.py`](file:///home/tan/04_Studies_Knowledge/UIT_Studies/Các môn học kỳ 3/CS114/run_experiments.py): Kịch bản chạy tự động 12 thực nghiệm và xuất toàn bộ kết quả.
* [`results/metrics_summary.csv`](file:///home/tan/04_Studies_Knowledge/UIT_Studies/Các môn học kỳ 3/CS114/results/metrics_summary.csv): Bảng tổng hợp số liệu 12 thực nghiệm.
* [`results/error_cases_analysis.csv`](file:///home/tan/04_Studies_Knowledge/UIT_Studies/Các môn học kỳ 3/CS114/results/error_cases_analysis.csv): 307 mẫu dự đoán sai phục vụ phân tích nguyên nhân.
* [`SUON_DO_AN_CS114.md`](file:///home/tan/04_Studies_Knowledge/UIT_Studies/Các môn học kỳ 3/CS114/SUON_DO_AN_CS114.md): Đề cương chi tiết và cấu trúc báo cáo 8 phần.
* [`README.md`](file:///home/tan/04_Studies_Knowledge/UIT_Studies/Các môn học kỳ 3/CS114/README.md): Báo cáo tóm tắt tổng quan đồ án trên GitHub.

# SƯỜN ĐỒ ÁN MÔN HỌC — CS114.MACHINE LEARNING
# ĐỀ TÀI: NHẬN DIỆN CẢM XÚC TRÊN MẠNG XÃ HỘI TIẾNG VIỆT (UIT-VSMEC)

> **Khoa**: Khoa học Máy tính — ĐH Công nghệ Thông tin (UIT, ĐHQG-HCM)  
> **Học phần**: CS114 — Nhập môn Máy học (Machine Learning)  
> **Bộ dữ liệu**: UIT-VSMEC (*Vietnamese Social Media Emotion Corpus*)  
> **Trạng thái**: Đã tải dữ liệu cục bộ (`CS114/data/vsmec/`) & hoàn thiện đề cương  
> **Nhóm thực hiện (4 thành viên)**:
> 1. **Lê Quang Thi** — MSSV: `25210337` (Nhóm trưởng)
> 2. **Trần Trọng Tấn** — MSSV: `25210334`
> 3. **Nguyễn Quang Lâm** — MSSV: `25210289`
> 4. **Võ Cẩm Thu** — MSSV: `25210342`

---

## 1. TỔNG QUAN & QUY ĐỊNH MÔN HỌC

| Mục | Chi tiết quy định từ giảng viên & đề cương |
|---|---|
| **Môn học** | CS114 — Nhập môn Máy học |
| **Quy mô nhóm** | 1 – 6 thành viên (tự chọn nhóm) |
| **Cơ cấu điểm** | **30%** Bài tập hàng tuần + **70%** Đồ án môn học *(Phương án lớp đông, miễn thi lý thuyết)* |
| **Báo cáo tiến độ** | 3 buổi báo cáo trực tiếp trong suốt kỳ học (bắt đầu từ Tuần 7–8) |
| **Hệ thống nộp bài** | Nộp qua WeCode UIT (`https://khmt.uit.edu.vn/wecode/cs114x/`) + GitHub Repository |
| **Yêu cầu thuật toán** | Cài đặt và so sánh **ít nhất 3 mô hình học máy cơ bản** (Scikit-learn) |
| **Độ đo bắt buộc** | Confusion Matrix, Accuracy, Precision, Recall, F1-Score (Macro), Phân tích Overfitting |

---

## 2. BỘ DỮ LIỆU CHUẨN: UIT-VSMEC

### 2.1 Nguồn gốc & Tính chính danh học thuật
- **Tên đầy đủ**: *UIT - Vietnamese Social Media Emotion Corpus*.
- **Đơn vị phát triển**: Nhóm nghiên cứu Xử lý Ngôn ngữ Tự nhiên (UIT NLP Group), ĐH Công nghệ Thông tin.
- **Công bố khoa học**: Xuất bản tại hội nghị quốc tế uy tín (IEEE/RIVF), là benchmark tiêu chuẩn cho bài toán phân loại cảm xúc tiếng Việt.
- **Ưu thế đặc biệt khi bảo vệ**: Sử dụng chính dữ liệu do UIT phát triển giúp đồ án có tính học thuật cao, trích dẫn chính xác tác giả UIT, tránh hoàn toàn rủi ro bị nghi ngờ về nguồn gốc hay tính hợp lệ của dữ liệu.

### 2.2 Quy mô & Cấu trúc phân chia (Benchmark Split)
Dữ liệu đã được chia sẵn thành 3 tập chuẩn (không bị rò rỉ dữ liệu — Data Leakage):

| Tập dữ liệu | Số lượng mẫu | Tỷ lệ | Đường dẫn lưu trữ |
|---|:---:|:---:|---|
| **Tập Huấn luyện (Train)** | 5.548 | 80,1% | `CS114/data/vsmec/train.csv` |
| **Tập Phát triển/Kiểm định (Validation)** | 686 | 9,9% | `CS114/data/vsmec/val.csv` |
| **Tập Kiểm thử (Test)** | 693 | 10,0% | `CS114/data/vsmec/test.csv` |
| **TỔNG CỘNG** | **6.927** | **100%** | Định dạng: `Sentence` (văn bản) & `Emotion` (nhãn) |

### 2.3 Phân bố nhãn cảm xúc & Hiện tượng mất cân bằng lớp (Class Imbalance)
Thống kê trực tiếp từ tập Train (5.548 câu):

| STT | Cảm xúc (Emotion) | Số lượng mẫu | Tỷ lệ (%) | Đặc điểm ngôn ngữ |
|:---:|---|:---:|:---:|---|
| 1 | **Enjoyment** (Thích thú/Vui vẻ) | 1.558 | 28,1% | Lớp đa số, chứa nhiều từ tích cực, emoji cười, từ khen ngợi |
| 2 | **Disgust** (Chán ghét/Kinh tởm) | 1.071 | 19,3% | Phổ biến trên mạng xã hội, từ ngữ bức xúc, châm biếm, phàn nàn |
| 3 | **Other** (Khác/Trung tính) | 1.021 | 18,4% | Câu hỏi, thông tin trung tính, không bộc lộ cảm xúc rõ ràng |
| 4 | **Sadness** (Buồn bã/Thất vọng) | 947 | 17,1% | Từ ngữ chia buồn, than vãn, thất bại, cô đơn |
| 5 | **Anger** (Tức giận/Phẫn nộ) | 391 | 7,0% | Từ ngữ kích động, chửi bới, phản đối gay gắt |
| 6 | **Fear** (Lo sợ/Bất an) | 318 | 5,7% | Bày tỏ nỗi sợ, hoang mang, cảnh báo nguy hiểm |
| 7 | **Surprise** (Ngạc nhiên/Bất ngờ) | 242 | 4,4% | Lớp thiểu số nhất, thán từ cảm thán, dấu chấm than, ngỡ ngàng |

> 💡 **Ý nghĩa thực nghiệm cho CS114 (Buổi 05)**: Sự chênh lệch giữa `Enjoyment` (28,1%) và `Surprise` (4,4%) là minh chứng thực tế rõ ràng để giải thích **vì sao Accuracy là thước đo lừa dối**, từ đó làm nổi bật sự cần thiết của **Macro-averaged F1-Score** và **Balanced Confusion Matrix**.

---

## 3. MỤC TIÊU & CÂU HỎI NGHIÊN CỨU (RESEARCH QUESTIONS)

### Câu hỏi nghiên cứu chính:
> *"Mô hình học máy cổ điển nào và phương pháp biểu diễn văn bản nào đạt hiệu năng tối ưu và kiểm soát quá khớp (overfitting) tốt nhất trên dữ liệu cảm xúc mạng xã hội tiếng Việt có độ mất cân bằng lớp cao?"*

### Ba câu hỏi nghiên cứu cụ thể:
- **RQ1 (Feature Representation)**: Kỹ thuật trích xuất đặc trưng nào mang lại biểu diễn ngữ nghĩa tốt hơn cho tiếng Việt: *Word-level TF-IDF* (từ đơn/ghép), *Character n-gram TF-IDF* (tiểu tự), hay *Mô hình lai kết hợp (Hybrid FeatureUnion)*?
- **RQ2 (Model Comparison & Multiclass Strategy)**: Trong các họ thuật toán *Probabilistic (Naive Bayes)*, *Linear/Margin-based (Logistic Regression, Linear SVM)* và *Ensemble (Random Forest/LightGBM)*, mô hình nào đạt Macro-F1 cao nhất khi giải bài toán 7 lớp với chiến lược *One-vs-Rest (OvR)* và *One-vs-One (OvO)*?
- **RQ3 (Generalization & Bias-Variance Tradeoff)**: Mức độ phức tạp của mô hình (thông qua siêu tham số điều chuẩn $C$ và độ sâu cây `max_depth`) ảnh hưởng như thế nào đến độ chênh lệch hiệu năng giữa tập Huấn luyện (Train) và tập Kiểm thử (Validation/Test)? Dấu hiệu Overfitting và Underfitting xuất hiện ở đâu?

---

## 4. MA TRẬN THỰC NGHIỆM 2 CHIỀU (2D EXPERIMENT DESIGN)

Để tạo điểm nhấn phương pháp luận vượt trội so với các đồ án thông thường, nhóm triển khai thực nghiệm theo ma trận 2 trục:

$$\text{3 Biểu diễn đặc trưng} \times \text{4 Mô hình học máy} = \mathbf{12\ Thí\ nghiệm\ đối\ sánh}$$

```
                          TRỤC ĐẶC TRƯNG (REPRESENTATION)
                   [Exp A]                  [Exp B]                  [Exp C]
               Word-level TF-IDF      Character n-gram TF-IDF       Hybrid (Word+Char)
              (1-gram, 2-gram)         (char n-grams: 2..5)           FeatureUnion
TRỤC MÔ HÌNH ┌──────────────────────┬──────────────────────┬──────────────────────┐
1. Naive Bayes│  MNB + Word (Base)   │      MNB + Char      │     MNB + Hybrid     │
2. Logistic   │  LR + Word (L2 reg)  │      LR + Char       │     LR + Hybrid      │
3. Linear SVM │  SVM + Word (Linear) │      SVM + Char      │     SVM + Hybrid     │
4. Ensemble   │  RF / LightGBM       │    RF / LightGBM     │    RF / LightGBM     │
              └──────────────────────┴──────────────────────┴──────────────────────┘
```

---

## 5. QUY TRÌNH THỰC HIỆN CHI TIẾT (PIPELINE)

### Bước 1 — Phân tích Khám phá Dữ liệu (EDA)
- **Thống kê mô tả**: Phân bố độ dài câu (theo số từ và số ký tự) theo từng nhãn cảm xúc.
- **Phát hiện ngoại lệ**: Xác định các câu quá ngắn ($< 2$ từ) hoặc các câu lặp ký tự (*"quáaaaa", "buồnnnnn"*).
- **Trực quan hóa**:
  - Biểu đồ cột phân bố tần suất 7 lớp cảm xúc (Seaborn Barplot).
  - Biểu đồ Boxplot phân bố độ dài câu theo từng nhóm cảm xúc.
  - Word Cloud các từ vựng đặc trưng nhất cho từng loại cảm xúc sau khi lọc từ dừng.

### Bước 2 — Tiền xử lý Văn bản Tiếng Việt
- **Làm sạch văn bản thô (Text Cleaning)**:
  - Chuyển toàn bộ về chữ thường (lowercase).
  - Chuẩn hóa mã bảng mã Unicode (NFC).
  - Xử lý teencode/từ lóng phổ biến trên mạng xã hội (*"ko" $\rightarrow$ "không", "dc" $\rightarrow$ "được", "bt" $\rightarrow$ "biết"*).
  - Chuẩn hóa kéo dài ký tự (*"vuiiiii quá"* $\rightarrow$ *"vui quá"*).
  - Xử lý Emoji: Giữ lại hoặc ánh xạ emoji thành token cảm xúc tương ứng (*😊 $\rightarrow$ "icon_vui", 😡 $\rightarrow$ "icon_gian"*).
- **Phân đoạn từ tiếng Việt (Word Segmentation)**:
  - Sử dụng thư viện chuyên dụng `underthesea` hoặc `pyvi` để ghép các từ phức tiếng Việt (*"học_sinh", "thất_vọng", "tức_giận"* thay vì tách rời từng âm tiết).
- **Xử lý từ dừng (Stopwords)**:
  - Loại bỏ các hư từ không mang sắc thái cảm xúc (*"thì", "là", "mà", "ở", "tại"*), nhưng **giữ lại các từ phủ định** (*"không", "chẳng", "chưa"*) vì chúng làm đảo ngược hoàn toàn cảm xúc câu.

### Bước 3 — Trích xuất Đặc trưng (Feature Extraction)
1. **Biểu diễn A (Word TF-IDF)**:
   - N-gram: $(1, 2)$ (từ đơn và từ đôi).
   - `min_df=3` (loại bỏ từ hiếm chỉ xuất hiện dưới 3 lần).
   - `max_features=3.000` (giới hạn tránh bùng nổ số chiều).
2. **Biểu diễn B (Character n-gram TF-IDF)**:
   - N-gram khoảng: $(2, 5)$ ký tự.
   - `analyzer='char'`.
   - `max_features=5.000`.
   - *Ưu thế*: Rất mạnh trong việc bắt lỗi chính tả, biến thể teencode trên mạng xã hội.
3. **Biểu diễn C (Hybrid FeatureUnion)**:
   - Kết hợp đồng thời Word TF-IDF và Char TF-IDF để tận dụng ngữ nghĩa từ ghép lẫn đặc trưng hình thái ký tự.

### Bước 4 — Cài đặt & Huấn luyện Mô hình Học máy
Triển khai 4 họ mô hình cốt lõi theo đúng chương trình học:
1. **Multinomial Naive Bayes (MNB)**:
   - Baseline chuẩn cho phân loại văn bản. Tinh chỉnh hệ số làm mềm Laplace $\alpha \in [0.1, 0.5, 1.0, 2.0]$.
2. **Logistic Regression (Softmax Regression / OvR)**:
   - Mô hình phân loại tuyến tính với hàm mất mát Cross-Entropy.
   - Tinh chỉnh tham số phạt điều chuẩn $C \in [0.01, 0.1, 1.0, 10.0, 100.0]$ với $L_2$ regularization.
3. **Support Vector Machine (Linear SVM)**:
   - Cực kỳ tối ưu cho không gian đặc trưng thưa và số chiều lớn như TF-IDF.
   - Sử dụng `LinearSVC` với hàm mất mát Hinge Loss hoặc Squared Hinge Loss.
4. **Học kết hợp — Ensemble (Random Forest / LightGBM)**:
   - Cài đặt `RandomForestClassifier` hoặc `LGBMClassifier` để làm đối trọng đại diện cho họ mô hình phi tuyến tính.
   - Phân tích sự khác biệt về hiệu năng và tốc độ tính toán khi xử lý ma trận thưa.

### Bước 5 — Chiến lược Phân loại Đa lớp (Multiclass Strategy)
Áp dụng trực tiếp kiến thức **Buổi 04** để so sánh:
- **One-vs-Rest (OvR)**: Huấn luyện $7$ bộ phân loại nhị phân độc lập.
- **One-vs-One (OvO)**: Huấn luyện $\frac{7 \times 6}{2} = 21$ bộ phân loại cặp.
- So sánh thời gian huấn luyện và độ chính xác phân tách giữa hai chiến lược.

### Bước 6 — Đánh giá Toàn diện & Phân tích Sai số
- **Bảng tổng hợp chỉ số**:
  - Accuracy (Độ chính xác tổng thể).
  - Macro-Precision & Macro-Recall (đảm bảo quyền lợi các lớp thiểu số).
  - **Macro-F1 Score** (chỉ số quyết định chính).
  - Weighted-F1 Score (để tham khảo tương quan dung lượng lớp).
- **Confusion Matrix Heatmap (7 × 7)**:
  - Trực quan hóa ma trận nhầm lẫn bằng Seaborn.
  - Phân tích các cặp cảm xúc hay bị dự đoán nhầm: ví dụ `Anger` thường bị nhầm sang `Disgust`, hoặc `Fear` bị nhầm sang `Surprise`. Giải thích nguyên nhân dưới góc độ ngữ nghĩa tiếng Việt.

### Bước 7 — Phân tích Quá khớp (Overfitting / Underfitting Analysis)
Áp dụng sâu sắc kiến thức **Buổi 05 (Bias-Variance Tradeoff)**:
1. **Đường cong học tập (Learning Curve)**:
   - Huấn luyện mô hình tốt nhất với các tỷ lệ tập dữ liệu tăng dần: $20\%, 40\%, 60\%, 80\%, 100\%$.
   - Vẽ đồ thị so sánh Training Score và Validation Score.
   - Phân tích độ dốc và khoảng cách (Gap) giữa hai đường để chứng minh tập dữ liệu 5.548 mẫu đã đủ để mô hình hội tụ hay còn cần thêm dữ liệu.
2. **Khảo sát tham số điều chuẩn ($C$)**:
   - Vẽ đồ thị F1-Score của tập Train và tập Val theo sự biến thiên của $\log(C)$.
   - Chỉ ra vùng $C \le 0.01$ gây ra **Underfitting** (Train và Val đều thấp).
   - Chỉ ra vùng $C \ge 100$ gây ra **Overfitting** (Train đạt gần $100\%$, Val suy giảm mạnh).
   - Xác định điểm cực trị $C$ tối ưu hóa sự cân bằng Bias - Variance.

---

## 6. CẤU TRÚC BÁO CÁO ĐỒ ÁN (WORD < 15 TRANG)

Báo cáo được thiết kế theo đúng cấu trúc chuẩn học thuật của UIT:

```
Trang bìa: Tên trường, đề tài, giảng viên hướng dẫn, danh sách sinh viên thực hiện
Phần 1: TÓM TẮT ĐỒ ÁN (Abstract - 0.5 trang)
  - Đặt vấn đề, mục tiêu, bộ dữ liệu UIT-VSMEC, kết quả nổi bật nhất.
Phần 2: GIỚI THIỆU & CÂU HỎI NGHIÊN CỨU (Introduction - 1.5 trang)
  - Tính cấp thiết của phân tích cảm xúc mạng xã hội tiếng Việt.
  - Phát biểu 3 câu hỏi nghiên cứu (RQ1, RQ2, RQ3).
Phần 3: KHÁM PHÁ DỮ LIỆU & TIỀN XỬ LÝ (EDA & Preprocessing - 2.5 trang)
  - Đặc tính tập dữ liệu UIT-VSMEC, phân bố 7 nhãn, biểu đồ EDA.
  - Kỹ thuật tiền xử lý văn bản tiếng Việt (tách từ underthesea, làm sạch).
Phần 4: PHƯƠNG PHÁP & TRÍCH XUẤT ĐẶC TRƯNG (Methodology - 2.5 trang)
  - Biểu diễn TF-IDF: Word, Char n-grams, Hybrid.
  - Cơ sở lý thuyết của 4 mô hình: Naive Bayes, Logistic, Linear SVM, Random Forest.
  - Chiến lược phân loại đa lớp: OvR vs OvO.
Phần 5: KẾT QUẢ THỰC NGHIỆM & ĐỐI SÁNH (Results - 3.0 trang)
  - Bảng so sánh 12 cấu hình thực nghiệm trên các độ đo (Accuracy, Macro-F1).
  - Phân tích chi tiết ma trận nhầm lẫn 7x7 (Confusion Matrix Analysis).
Phần 6: PHÂN TÍCH QUÁ KHỚP & ĐÁNH ĐỔI BIAS - VARIANCE (Analysis - 2.5 trang)
  - Đồ thị Learning Curve và phân tích hội tụ.
  - Đồ thị Regularization Path và phân tích điểm tối ưu.
Phần 7: KẾT LUẬN & HƯỚNG PHÁT TRIỂN (Conclusion - 1.0 trang)
  - Trả lời trọn vẹn 3 Research Questions.
  - Hạn chế của mô hình ML cổ điển và hướng mở rộng (PhoBERT, Deep Learning).
Phần 8: TÀI LIỆU THAM KHẢO & PHỤ LỤC (References - 1.5 trang)
  - Trích dẫn bài báo gốc UIT-VSMEC, tài liệu giảng dạy CS114, link GitHub repo.
```

---

## 7. KIẾN TRÚC MÃ NGUỒN (PROJECT STRUCTURE)

Cấu trúc thư mục được quy hoạch chuẩn mực, sẵn sàng đẩy lên GitHub:

```
CS114_UIT_VSMEC_Emotion_Recognition/
│
├── data/
│   └── vsmec/
│       ├── train.csv              # 5,548 mẫu huấn luyện (đã tải)
│       ├── val.csv                # 686 mẫu kiểm định (đã tải)
│       └── test.csv               # 693 mẫu kiểm thử độc lập (đã tải)
│
├── notebooks/
│   ├── 01_eda_and_visualization.ipynb      # EDA, biểu đồ phân bố, wordcloud
│   ├── 02_text_preprocessing.ipynb         # Làm sạch, tách từ underthesea
│   ├── 03_feature_extraction_tfidf.ipynb   # Word vs Char vs Hybrid TF-IDF
│   ├── 04_model_training_comparison.ipynb  # Huấn luyện 4 mô hình, OvR vs OvO
│   └── 05_overfitting_evaluation.ipynb     # Learning curves, tuning C, Confusion Matrix
│
├── src/
│   ├── __init__.py
│   ├── preprocessor.py            # Hàm xử lý regex, teencode, underthesea
│   ├── feature_builder.py         # Pipeline trích xuất đặc trưng TF-IDF
│   ├── models.py                  # Khởi tạo và bọc các mô hình ML
│   ├── evaluate.py                # Tính toán classification report, Macro-F1
│   └── plot_utils.py              # Vẽ heatmap, learning curve, regularization path
│
├── results/
│   ├── figures/                   # Lưu trữ biểu đồ PNG độ phân giải cao
│   │   ├── class_distribution.png
│   │   ├── confusion_matrix_best_model.png
│   │   ├── learning_curve.png
│   │   └── regularization_tuning_curve.png
│   ├── metrics_summary.csv        # Bảng kết quả tổng hợp 12 thực nghiệm
│   └── error_cases_analysis.csv   # Danh sách các câu bị dự đoán sai nhiều nhất
│
├── report/
│   └── Bao_Cao_Do_An_CS114_UIT_VSMEC.docx  # Báo cáo chính thức nộp chấm điểm
│
├── requirements.txt               # Thư viện: scikit-learn, underthesea, pandas, seaborn...
└── README.md                      # Hướng dẫn chạy code và tóm tắt kết quả
```

---

## 8. CHECKLIST THỰC HIỆN TỪNG BƯỚC

- [x] **Bước 1**: Tải và xác thực bộ dữ liệu UIT-VSMEC cục bộ (đạt 6.927 mẫu).
- [x] **Bước 2**: Hoàn thiện Đề cương & Sườn đồ án chuẩn học thuật UIT (`SUON_DO_AN_CS114.md`).
- [ ] **Bước 3**: Cài đặt môi trường (`requirements.txt`: `scikit-learn`, `underthesea`, `seaborn`, `pandas`).
- [ ] **Bước 4**: Xây dựng module tiền xử lý tiếng Việt `src/preprocessor.py`.
- [ ] **Bước 5**: Chạy thử nghiệm trích xuất 3 dạng TF-IDF `src/feature_builder.py`.
- [ ] **Bước 6**: Huấn luyện ma trận 12 thực nghiệm, xuất kết quả `results/metrics_summary.csv`.
- [ ] **Bước 7**: Xuất biểu đồ Confusion Matrix, Learning Curves, C-tuning vào `results/figures/`.
- [ ] **Bước 8**: Viết báo cáo Word chính thức theo cấu trúc 8 phần (< 15 trang).
- [ ] **Bước 9**: Soạn Slide thuyết trình tóm tắt và đồng bộ toàn bộ mã nguồn lên GitHub.

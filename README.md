# CS114.F31.CN2.TTNT — ĐỒ ÁN MÔN HỌC MÁY HỌC (MACHINE LEARNING)
## Đề tài: Nhận diện Cảm xúc trên Mạng xã hội Tiếng Việt (UIT-VSMEC Emotion Recognition)

> **Trường**: Đại học Công nghệ Thông tin — ĐHQG-HCM (UIT)  
> **Mã lớp**: CS114.F31.CN2.TTNT — Nhập môn Máy học  
> **Sinh viên thực hiện**: Trần Trọng Tấn (25210334)  
> **Bộ dữ liệu**: UIT-VSMEC (*Vietnamese Social Media Emotion Corpus*) — 6.927 mẫu, 7 nhãn  

---

## 1. TỔNG QUAN ĐỀ TÀI

Đồ án tập trung nghiên cứu bài toán **Phân loại cảm xúc đa lớp (Multiclass Emotion Classification)** trên văn bản mạng xã hội tiếng Việt bằng các phương pháp học máy cổ điển (*Classical Machine Learning*).

### Đặc điểm bộ dữ liệu chuẩn UIT-VSMEC:
- **Nguồn gốc**: Nhóm nghiên cứu Xử lý Ngôn ngữ Tự nhiên (UIT NLP Group), ĐH Công nghệ Thông tin.
- **Quy mô**: 6.927 câu bình luận mạng xã hội, chia sẵn theo chuẩn benchmark:
  - **Train**: 5.548 câu (80,1%)
  - **Validation**: 686 câu (9,9%)
  - **Test**: 693 câu (10,0%)
- **7 nhãn cảm xúc**: `Enjoyment` (28,1%), `Disgust` (19,3%), `Other` (18,4%), `Sadness` (17,1%), `Anger` (7,0%), `Fear` (5,7%), `Surprise` (4,4%).
- **Thách thức**: Văn bản mạng xã hội chứa nhiều teencode, từ lóng, emoji, lỗi chính tả và **hiện tượng mất cân bằng lớp nghiêm trọng (Class Imbalance)**.

---

## 2. THIẾT KẾ THỰC NGHIỆM 2 CHIỀU (3 × 4 = 12 CẤU HÌNH)

Hệ thống được thiết kế theo ma trận 2 trục đối sánh độc lập:

1. **Trục biểu diễn đặc trưng (3 Representations)**:
   - `Word_TFIDF`: Word-level n-grams (1, 2), `max_features=3000`, bắt ngữ nghĩa từ ghép tiếng Việt.
   - `Char_TFIDF`: Character-level n-grams (2..5), `max_features=5000`, bắt biến thể chính tả, teencode, icon.
   - `Hybrid_TFIDF`: Mô hình lai kết hợp `FeatureUnion` (6.000 chiều).

2. **Trục mô hình học máy (4 Models)**:
   - **Multinomial Naive Bayes (MNB)**: Mô hình xác suất baseline.
   - **Logistic Regression (LR)**: Mô hình tuyến tính phân loại đa lớp với hàm phạt $L_2$.
   - **Linear Support Vector Machine (Linear SVM)**: Mô hình tìm siêu phẳng phân tách cực đại.
   - **Random Forest (RF)**: Mô hình học kết hợp phi tuyến (Ensemble Bagging).

---

## 3. KẾT QUẢ THỰC NGHIỆM ĐỐI SÁNH

Bảng tổng hợp xếp hạng 12 cấu hình thực nghiệm trên tập Validation (sắp xếp theo **Macro-F1**):

| Hạng | Biểu diễn đặc trưng | Mô hình ML | Val Accuracy | Val Macro-F1 | Train-Val Gap | Đánh giá Overfitting |
|:---:|---|---|:---:|:---:|:---:|:---:|
| 🥇 | **Hybrid_TFIDF** | **LogisticRegression** | **0.5685** | **0.4917** | 0.2890 | Kiểm soát tốt nhất |
| 🥈 | Word_TFIDF | LinearSVM | 0.5379 | 0.4882 | 0.4357 | Overfitting nhẹ |
| 🥉 | Hybrid_TFIDF | LinearSVM | 0.5277 | 0.4796 | 0.4981 | Overfitting |
| 4 | Hybrid_TFIDF | MultinomialNB | 0.5423 | 0.4703 | 0.2005 | Bias cao (Underfit) |
| 5 | Char_TFIDF | LinearSVM | 0.5248 | 0.4598 | 0.4278 | Overfitting |
| 6 | Word_TFIDF | LogisticRegression | 0.5525 | 0.4521 | 0.2453 | Khá cân bằng |
| 7 | Char_TFIDF | LogisticRegression | 0.5394 | 0.4296 | 0.2271 | Khá cân bằng |
| 8 | Word_TFIDF | MultinomialNB | 0.5262 | 0.3954 | 0.2381 | Underfit |
| 9 | Char_TFIDF | MultinomialNB | 0.5000 | 0.3847 | 0.1866 | Underfit |
| 10 | Char_TFIDF | RandomForest | 0.4840 | 0.3600 | 0.5366 | Overfit nặng |
| 11 | Hybrid_TFIDF | RandomForest | 0.4650 | 0.3444 | 0.5613 | Overfit nặng |
| 12 | Word_TFIDF | RandomForest | 0.4738 | 0.3404 | 0.3227 | Hiệu năng thấp |

### Đánh giá chốt chặn trên Holdout Test Set (Mô hình tối ưu nhất: `Hybrid_TFIDF + LogisticRegression`):
- **Accuracy**: `0.5570` (55,70%)
- **Balanced Accuracy**: `0.4593` (45,93%)
- **Macro-Precision**: `0.5702` (57,02%)
- **Macro-Recall**: `0.4593` (45,93%)
- **⭐ Macro-F1 Score**: `0.4836` (48,36%)
- **Weighted-F1 Score**: `0.5466` (54,66%)

---

## 4. PHÂN TÍCH CHUYÊN SÂU THEO ĐỀ CƯƠNG CS114

### 4.1 Chiến lược phân loại đa lớp (Buổi 04): OvR vs OvO
Thực nghiệm so sánh trên mô hình Logistic Regression:
- **One-vs-Rest (OvR — 7 mô hình nhị phân)**: Thời gian huấn luyện $1.44\text{s}$ | **Val Acc = 0.5671** | **Val Macro-F1 = 0.4881**
- **One-vs-One (OvO — 21 mô hình cặp)**: Thời gian huấn luyện $0.85\text{s}$ | **Val Acc = 0.5525** | **Val Macro-F1 = 0.4503**
- $\rightarrow$ **Kết luận**: Chiến lược **One-vs-Rest (OvR)** vượt trội hơn OvO trên dữ liệu này nhờ tận dụng được toàn bộ mẫu dữ liệu huấn luyện cho mỗi bộ phân loại nhị phân.

### 4.2 Phân tích hiện tượng Quá khớp & Bias-Variance (Buổi 05)
- **Random Forest** gặp hiện tượng quá khớp nghiêm trọng (Train F1 gần $100\%$ nhưng Val F1 chỉ $\approx 34\%$, Gap $> 0.55$). Điều này chứng minh định đề lý thuyết: cây quyết định/ensemble dạng bagging không tối ưu cho không gian văn bản thưa thớt có số chiều lớn ($> 3.000$ chiều).
- **Logistic Regression & Linear SVM** kiểm soát phương sai tốt hơn nhờ cơ chế phạt điều chuẩn $L_2$ ($C=1.0$).
- Biểu đồ **Learning Curves** và **Regularization Path** được tự động xuất ra thư mục `results/figures/` để phục vụ làm báo cáo đồ án.

---

## 5. HƯỚNG DẪN CÀI ĐẶT & CHẠY THỰC NGHIỆM

### 5.1 Cài đặt môi trường
```bash
git clone https://github.com/Trantrongtan2000/CS114.F31.CN2.TTNT.git
cd CS114.F31.CN2.TTNT
pip install -r requirements.txt
```

### 5.2 Chạy toàn bộ pipeline thực nghiệm
```bash
python run_experiments.py
```
*Thời gian chạy toàn bộ: $\approx 1$ phút. Dữ liệu làm sạch được tự động cache để các lần chạy sau chỉ mất vài giây.*

---

## 6. CẤU TRÚC THƯ MỤC DỰ ÁN

```
CS114.F31.CN2.TTNT/
├── data/
│   └── vsmec/
│       ├── train.csv                      # 5.548 mẫu huấn luyện gốc
│       ├── val.csv                        # 686 mẫu kiểm định gốc
│       └── test.csv                       # 693 mẫu kiểm thử độc lập
│
├── src/
│   ├── __init__.py
│   ├── preprocessor.py                    # Làm sạch teencode, emoji, phân đoạn từ pyvi
│   ├── feature_builder.py                 # Xây dựng 3 dạng TF-IDF (Word, Char, Hybrid)
│   ├── models.py                          # 4 mô hình ML & bộ bọc OvR/OvO
│   ├── evaluate.py                        # Tính toán các độ đo & Train-Val Gap
│   └── plot_utils.py                      # Xuất biểu đồ heatmap, learning curve, C tuning
│
├── results/
│   ├── figures/                           # 5 biểu đồ trực quan hóa báo cáo
│   │   ├── 01_class_distribution.png
│   │   ├── 02_model_comparison.png
│   │   ├── 03_confusion_matrix_best_model.png
│   │   ├── 04_regularization_tuning_curve.png
│   │   └── 05_learning_curve_best_model.png
│   ├── metrics_summary.csv                # Bảng số liệu chi tiết 12 thực nghiệm
│   └── error_cases_analysis.csv           # 307 ca dự đoán sai phục vụ Error Analysis
│
├── SUON_DO_AN_CS114.md                    # Sườn đồ án chi tiết & Khung báo cáo < 15 trang
├── run_experiments.py                     # Script điều phối toàn bộ thực nghiệm
├── requirements.txt                       # Thư viện phụ thuộc
└── README.md                              # Báo cáo tổng kết đồ án
```

---

*Đồ án hoàn thành theo đúng đề cương và yêu cầu môn học CS114 — Trường ĐH Công nghệ Thông tin (UIT).*

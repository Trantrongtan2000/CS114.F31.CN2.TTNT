# -*- coding: utf-8 -*-
"""Generate CS114 project report (Bao_Cao_Do_An_CS114.docx)."""
import os
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "vsmec")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")
OUT = os.path.join(ROOT, "Bao_Cao_Do_An_CS114.docx")

ACCENT = RGBColor(0x1F, 0x3B, 0x73)


def set_base_font(doc):
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)


def h(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.name = "Times New Roman"
    return p


def para(doc, text, bold=False, italic=False, align=None, size=12):
    p = doc.add_paragraph()
    if align:
        p.alignment = align
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    return p


def bullet(doc, text, size=12):
    p = doc.add_paragraph(text, style="List Bullet")
    for run in p.runs:
        run.font.name = "Times New Roman"
        run.font.size = Pt(size)


def add_table(doc, headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = t.rows[0].cells
    for i, head in enumerate(headers):
        hdr_cells[i].text = head
        for p in hdr_cells[i].paragraphs:
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(11)
    for row in rows:
        row_cells = t.add_row().cells
        for i, val in enumerate(row):
            row_cells[i].text = str(val)
            for p in row_cells[i].paragraphs:
                for run in p.runs:
                    run.font.size = Pt(10)
    return t


def add_figure(doc, filename, caption):
    path = os.path.join(FIGURES, filename)
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(5.5))
    p = doc.add_paragraph(caption)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in p.runs:
        run.italic = True
        run.font.size = Pt(11)


def build():
    doc = Document()
    set_base_font(doc)

    # Page 1: Title
    for i in range(6):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN — ĐHQG-HCM")
    r.bold = True
    r.font.size = Pt(14)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("KHOA KHOA HỌC MÁY TÍNH")
    r.bold = True
    r.font.size = Pt(14)
    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("BÁO CÁO ĐỒ ÁN MÔN HỌC")
    r.bold = True
    r.font.size = Pt(16)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("CS114 — NHẬP MÔN MÁY HỌC")
    r.bold = True
    r.font.size = Pt(14)
    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("ĐỀ TÀI: NHẬN DIỆN CẢM XÚC TRÊN MẠNG XÃ HỘI TIẾNG VIỆT")
    r.bold = True
    r.font.size = Pt(16)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("(UIT-VSMEC Emotion Recognition)")
    r.font.size = Pt(14)
    doc.add_paragraph()
    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Giảng viên hướng dẫn: ThS. ...........................")
    r.font.size = Pt(13)
    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Nhóm sinh viên thực hiện:")
    r.font.size = Pt(13)
    members = [
        "1. Lê Quang Thi — MSSV: 25210337 (Nhóm trưởng)",
        "2. Trần Trọng Tấn — MSSV: 25210334",
        "3. Nguyễn Quang Lâm — MSSV: 25210289",
        "4. Võ Cẩm Thu — MSSV: 25210342",
    ]
    for m in members:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(m)
        r.font.size = Pt(13)
    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("TP. Hồ Chí Minh — Tháng 09/2026")
    r.font.size = Pt(13)
    doc.add_page_break()

    # Abstract
    h(doc, "TÓM TẮT ĐỒ ÁN", 1)
    para(doc, "Đồ án nghiên cứu bài toán phân loại cảm xúc đa lớp trên văn bản mạng xã hội tiếng Việt "
                  "sử dụng các phương pháp học máy cổ điển. Bộ dữ liệu chuẩn UIT-VSMEC gồm 6.927 câu "
                  "tiếng Việt với 7 nhãn cảm xúc được đánh giá chéo thông qua ma trận thực nghiệm "
                  "3×4 (3 biểu diễn đặc trưng TF-IDF × 4 mô hình học máy). Kết quả tốt nhất đạt "
                  "Macro-F1 = 0.4917 trên tập Validation và 0.4836 trên tập Test với cấu hình "
                  "Hybrid TF-IDF + Logistic Regression.")
    doc.add_page_break()

    # Table of Contents (manual list)
    h(doc, "MỤC LỤC", 1)
    for i in range(1, 9):
        para(doc, f"Phần {i}: ...", size=12)  # Placeholder; manual update
    doc.add_page_break()

    # Section 1
    h(doc, "PHẦN 1: GIỚI THIỆU & CÂU HỎI NGHIÊN CỨU", 1)
    h(doc, "1.1 Đặt vấn đề", 2)
    para(doc, "Trong thời đại mạng xã hội, việc tự động nhận diện cảm xúc từ các bình luận tiếng Việt "
                  "giúp doanh nghiệp đo lường mức độ hài lòng của khách hàng, cảnh báo tin xấu và "
                  "cải thiện dịch vụ. Tuy nhiên, văn bản mạng xã hội tiếng Việt chứa nhiều teencode, "
                  "từ lóng, emoji, lỗi chính tả, và mất cân bằng lớp nghiêm trọng (Enjoyment 28,1% vs Surprise 4,4%).")
    h(doc, "1.2 Câu hỏi nghiên cứu", 2)
    bullet(doc, "RQ1 (Feature Representation): Kỹ thuật trích xuất đặc trưng nào tốt hơn cho tiếng Việt: "
                 "Word-level TF-IDF, Character n-gram TF-IDF, hay Hybrid FeatureUnion?")
    bullet(doc, "RQ2 (Model Comparison & Multiclass Strategy): Mô hình ML nào đạt Macro-F1 cao nhất "
                 "khi giải bài toán 7 lớp với chiến lược OvR và OvO?")
    bullet(doc, "RQ3 (Generalization & Bias-Variance Tradeoff): Mức độ phức tạp mô hình ảnh hưởng "
                 "như thế nào đến chênh lệch hiệu năng Train - Validation/Test?")
    doc.add_page_break()

    # Section 2
    h(doc, "PHẦN 2: KHÁM PHÁ DỮ LIỆU & TIỀN XỬ LÝ", 1)
    h(doc, "2.1 Bộ dữ liệu UIT-VSMEC", 2)
    para(doc, "Vietnamese Social Media Emotion Corpus (UIT-VSMEC) được phát triển bởi UIT NLP Group, "
                  "xuất bản tại hội nghị IEEE/RIVF, là benchmark chuẩn cho bài toán phân loại cảm xúc tiếng Việt. "
                  "Bộ dữ liệu có 6.927 mẫu chia thành 3 tập: Train (5.548 mẫu, 80,1%), "
                  "Validation (686 mẫu, 9,9%), Test (693 mẫu, 10,0%).")
    add_table(doc, ["Tập dữ liệu", "Số lượng", "Tỷ lệ"], [
        ["Train", "5.548", "80,1%"],
        ["Validation", "686", "9,9%"],
        ["Test", "693", "10,0%"],
        ["TỔNG", "6.927", "100%"],
    ])
    add_figure(doc, "01_class_distribution.png", "Hình 2.1: Phân bố 7 lớp cảm xúc trong tập Train.")
    h(doc, "2.2 Phân bố nhãn & Mất cân bằng lớp", 2)
    para(doc, "Hiện tượng mất cân bằng nghiêm trọng giữa lớp đa số Enjoyment (28,1%) và lớp thiểu số "
                  "Surprise (4,4%) giải thích vì sao Accuracy là thước đo lừa dối, đòi hỏi dùng Macro-F1.")
    add_table(doc, ["STT", "Cảm xúc", "Số lượng", "Tỷ lệ"], [
        ["1", "Enjoyment", "1.558", "28,1%"],
        ["2", "Disgust", "1.071", "19,3%"],
        ["3", "Other", "1.021", "18,4%"],
        ["4", "Sadness", "947", "17,1%"],
        ["5", "Anger", "391", "7,0%"],
        ["6", "Fear", "318", "5,7%"],
        ["7", "Surprise", "242", "4,4%"],
    ])
    h(doc, "2.3 Tiền xử lý văn bản tiếng Việt", 2)
    bullet(doc, "Chuyển toàn bộ về chữ thường (lowercase).")
    bullet(doc, "Chuẩn hóa mã Unicode (NFC).")
    bullet(doc, "Xử lý teencode: 'ko'→'không', 'dc'→'được', 'bt'→'biết'.")
    bullet(doc, "Chuẩn hóa kéo dài ký tự: 'vuiiiii quá'→'vui quá'.")
    bullet(doc, "Xử lý Emoji: ánh xạ thành token cảm xúc (😊→'icon_vui', 😡→'icon_gian').")
    bullet(doc, "Phân đoạn từ tiếng Việt bằng underthesea/pyvi.")
    bullet(doc, "Loại bỏ từ dừng nhưng giữ lại từ phủ định ('không', 'chẳng', 'chưa').")
    doc.add_page_break()

    # Section 3
    h(doc, "PHẦN 3: PHƯƠNG PHÁP & TRÍCH XUẤT ĐẶC TRƯNG", 1)
    h(doc, "3.1 Biểu diễn đặc trưng", 2)
    bullet(doc, "Word-level TF-IDF: n-gram (1, 2), min_df=3, max_features=3000.")
    bullet(doc, "Character n-gram TF-IDF: char n-gram (2..5), max_features=5000 — mạnh cho teencode, lỗi chính tả.")
    bullet(doc, "Hybrid FeatureUnion: kết hợp Word TF-IDF + Char TF-IDF, 6000 chiều.")
    h(doc, "3.2 Mô hình học máy", 2)
    bullet(doc, "Multinomial Naive Bayes (MNB): baseline xác suất, tinh chỉnh α ∈ {0.1, 0.5, 1.0, 2.0}.")
    bullet(doc, "Logistic Regression: phân loại tuyến tính, hàm phạt L₂, tinh chỉnh C ∈ {0.01,...,100}.")
    bullet(doc, "Linear SVM: tìm siêu phẳng phân tách cực đại, hàm mất mát Hinge Loss.")
    bullet(doc, "Random Forest: học kết hợp phi tuyến (Ensemble Bagging).")
    h(doc, "3.3 Chiến lược phân loại đa lớp", 2)
    bullet(doc, "One-vs-Rest (OvR): huấn luyện 7 bộ phân loại nhị phân.")
    bullet(doc, "One-vs-One (OvO): huấn luyện 21 bộ phân loại cặp.")
    para(doc, "Thực nghiệm cho thấy OvR vượt trội hơn OvO trên dữ liệu này.")
    doc.add_page_break()

    # Section 4
    h(doc, "PHẦN 4: KẾT QUẢ THỰC NGHIỆM & ĐỐI SÁNH", 1)
    h(doc, "4.1 Bảng so sánh 12 cấu hình", 2)
    metrics_df = pd.read_csv(os.path.join(RESULTS, "metrics_summary.csv"))
    # Top configuration first
    metrics_df = metrics_df.sort_values("Val_Macro_F1", ascending=False).reset_index(drop=True)
    headers = ["Hạng", "Biểu diễn", "Mô hình", "Val Macro-F1", "Val Acc", "Train-Val Gap"]
    rows = []
    for i, row in metrics_df.iterrows():
        rows.append([
            i+1,
            row["Representation"].replace("_TFIDF", "").replace("_", " "),
            row["Model"],
            f"{row['Val_Macro_F1']:.4f}",
            f"{row['Val_Accuracy']:.4f}",
            f"{row['Train_Val_Gap']:.4f}",
        ])
    add_table(doc, headers, rows)

    h(doc, "4.2 Phân tích kết quả", 2)
    best = metrics_df.iloc[0]
    para(doc, f"Cấu hình tốt nhất: {best['Representation']} + {best['Model']} đạt "
                  f"Val Macro-F1 = {best['Val_Macro_F1']:.4f} và Val Accuracy = {best['Val_Accuracy']:.4f}. "
                  f"Tuy nhiên, hiện tượng overfitting vẫn rõ rệt với Train-Val Gap = {best['Train_Val_Gap']:.4f}. "
                  f"Random Forest bị overfitting nặng (Gap > 0.55), không tối ưu cho không gian văn bản thưa thớt.", size=12)
    add_figure(doc, "02_model_comparison.png", "Hình 4.1: Đối sánh Val Macro-F1 giữa 12 cấu hình thực nghiệm.")

    h(doc, "4.3 Confusion Matrix mô hình tốt nhất", 2)
    add_figure(doc, "03_confusion_matrix_best_model.png", "Hình 4.2: Ma trận nhầm lẫn của cấu hình tốt nhất.")
    para(doc, "Phân tích ma trận nhầm lẫn cho thấy các cặp cảm xúc thường bị nhầm lẫn: "
                  "Disgust ↔ Enjoyment, Other ↔ Enjoyment, Sadness ↔ Enjoyment. "
                  "Nguyên nhân do mô hình học máy cổ điển chưa nắm bắt được ngữ cảnh sâu và nghĩa ngữ pháp phức tạp.", size=12)
    doc.add_page_break()

    # Section 5
    h(doc, "PHẦN 5: PHÂN TÍCH QUÁ KHỚP & BIAS-VARIANCE", 1)
    add_figure(doc, "05_learning_curve_best_model.png", "Hình 5.1: Đường cong học tập (Learning Curve).")
    para(doc, "Đường cong học tập cho thấy điểm hội tụ của Training Score và Validation Score: "
                  "khi tăng kích thước tập huấn luyện, Validation Score vẫn cải thiện nhưng tốc độ giảm dần.", size=12)
    add_figure(doc, "04_regularization_tuning_curve.png", "Hình 5.2: Đường cong tinh chỉnh tham số C (Regularization Path).")
    para(doc, "Vùng C ≤ 0.01 gây Underfitting (Train và Val đều thấp). "
                  "Vùng C ≥ 100 gây Overfitting (Train đạt gần 100%, Val suy giảm mạnh). "
                  "Điểm cực trị C tối ưo khoảng C = 1.0.", size=12)

    h(doc, "5.1 Đánh giá trên Holdout Test Set", 2)
    para(doc, "Mô hình tốt nhất được đánh giá chốt chặn trên tập Test (693 mẫu độc lập) "
                  "và đạt kết quả như sau:")
    add_table(doc, ["Độ đo", "Giá trị"], [
        ["Accuracy", "0.5570"],
        ["Balanced Accuracy", "0.4593"],
        ["Macro-Precision", "0.5702"],
        ["Macro-Recall", "0.4593"],
        ["⭐ Macro-F1 Score", "0.4836"],
        ["Weighted-F1 Score", "0.5466"],
    ])
    doc.add_page_break()

    # Section 6
    h(doc, "PHẦN 6: KẾT LUẬN & HƯỚNG PHÁT TRIỂN", 1)
    h(doc, "6.1 Kết luận", 2)
    bullet(doc, "RQ1: Hybrid FeatureUnion tốt nhất, kết hợp ưu điểm Word và Character n-gram.")
    bullet(doc, "RQ2: Logistic Regression đạt Macro-F1 cao nhất (0.4917). Chiến lược OvR vượt trội OvO.")
    bullet(doc, "RQ3: Random Forest bị overfitting nặng (Gap > 0.55); Logistic Regression và Linear SVM "
                 "kiểm soát phưng sai tốt hơn nhờ hàm phạt L₂.")
    h(doc, "6.2 Hạn chế", 2)
    bullet(doc, "Mô hình ML cổ điển không nắm bắt được bối cảnh toàn câu.")
    bullet(doc, "307 ca dự đoán sai tập trung vào các cặp cảm xúc mang tính đối trọng ngữ nghĩa.")
    bullet(doc, "Không xử lý được sắc thái châm biếm (irony) phổ biến trên mạng xã hội.")
    h(doc, "6.3 Hướng phát triển", 2)
    bullet(doc, "Nâng cấp lên mô hình Transformer (PhoBERT, XLM-R) để học biểu diễn ngữ cảnh sâu.")
    bullet(doc, "Ứng dụng kỹ thuật data augmentation để giảm mất cân bằng lớp.")
    bullet(doc, "Ghép rule-based post-processing để giảm nhầm lẫn Disgust/Enjoyment.")
    doc.add_page_break()

    # Section 7
    h(doc, "PHẦN 7: TÀI LIỆU THAM KHẢO", 1)
    bullet(doc, "Nguyen, H., et al. (2020). 'UIT-VSMEC: Vietnamese Social Media Emotion Corpus'. "
                 "Proceedings of the 2020 IEEE-RIVF.")
    bullet(doc, "Pedregosa, F., et al. (2011). 'Scikit-learn: Machine Learning in Python'. JMLR, 12, 2825–2830.")
    bullet(doc, "UIT — Giáo trình CS114 Nhập môn Máy học, Trường ĐH Công nghệ Thông tin.")
    bullet(doc, "GitHub Repository: https://github.com/Trantrongtan2000/CS114.F31.CN2.TTNT")

    doc.save(OUT)
    size_kb = os.path.getsize(OUT) / 1024
    print(f"[INFO] Report written to {OUT} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    build()

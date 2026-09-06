# -*- coding: utf-8 -*-
"""
Script: run_experiments.py
Chức năng: Điều phối toàn bộ quy trình thực nghiệm Đồ án CS114 (UIT-VSMEC):
  1. Tải và tiền xử lý dữ liệu (có caching để chạy siêu nhanh ở các lần sau)
  2. Vẽ biểu đồ phân bố lớp cảm xúc (Class Imbalance)
  3. Chạy ma trận 12 thực nghiệm (3 Biểu diễn đặc trưng x 4 Mô hình)
  4. Phân tích hiện tượng Overfitting / Underfitting (Train-Val Gap)
  5. Thử nghiệm chiến lược đa lớp: One-vs-Rest (OvR) vs One-vs-One (OvO)
  6. Lựa chọn mô hình tốt nhất -> Đánh giá chốt chặn trên Holdout Test Set
  7. Xuất toàn bộ biểu đồ (Confusion Matrix, Learning Curve, C-tuning, Model Comparison)
  8. Xuất bảng tổng hợp metrics.csv và file phân tích ca dự đoán sai (Error Analysis)
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Thêm thư mục hiện tại vào Python Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.preprocessor import batch_preprocess
from src.feature_builder import get_all_vectorizers
from src.models import get_base_models, wrap_multiclass_strategy
from src.evaluate import evaluate_predictions, compute_generalization_gap
from src.plot_utils import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_learning_curve_graph,
    plot_regularization_path,
    plot_experiment_comparison
)


DATA_DIR = os.path.join(BASE_DIR, "data", "vsmec")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)


def load_and_preprocess_data():
    """Tải dữ liệu và thực hiện tiền xử lý văn bản tiếng Việt (có caching)."""
    train_clean_path = os.path.join(DATA_DIR, "train_cleaned.csv")
    val_clean_path = os.path.join(DATA_DIR, "val_cleaned.csv")
    test_clean_path = os.path.join(DATA_DIR, "test_cleaned.csv")

    if os.path.exists(train_clean_path) and os.path.exists(val_clean_path) and os.path.exists(test_clean_path):
        print("📂 Đang tải dữ liệu đã tiền xử lý từ cache...")
        train_df = pd.read_csv(train_clean_path)
        val_df = pd.read_csv(val_clean_path)
        test_df = pd.read_csv(test_clean_path)
    else:
        print("⚙️  Bắt đầu quy trình tiền xử lý văn bản tiếng Việt UIT-VSMEC...")
        train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
        test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

        print("  - Xử lý tập Train (5,548 câu)...")
        train_df["Cleaned_Sentence"] = batch_preprocess(train_df["Sentence"].tolist())
        print("  - Xử lý tập Val (686 câu)...")
        val_df["Cleaned_Sentence"] = batch_preprocess(val_df["Sentence"].tolist())
        print("  - Xử lý tập Test (693 câu)...")
        test_df["Cleaned_Sentence"] = batch_preprocess(test_df["Sentence"].tolist())

        # Lưu cache
        train_df.to_csv(train_clean_path, index=False, encoding="utf-8")
        val_df.to_csv(val_clean_path, index=False, encoding="utf-8")
        test_df.to_csv(test_clean_path, index=False, encoding="utf-8")
        print("💾 Đã lưu dữ liệu làm sạch vào cache.")

    return train_df, val_df, test_df


def main():
    print("=" * 70)
    print("🚀 ĐỒ ÁN CS114: NHẬN DIỆN CẢM XÚC TIẾNG VIỆT (UIT-VSMEC)")
    print("=" * 70)

    # 1. Tải và tiền xử lý
    train_df, val_df, test_df = load_and_preprocess_data()
    labels = sorted(train_df["Emotion"].unique().tolist())
    print(f"📊 Tập nhãn cảm xúc ({len(labels)} lớp): {labels}")

    # 2. Vẽ biểu đồ phân bố lớp
    plot_class_distribution(
        train_df["Emotion"],
        os.path.join(FIGURES_DIR, "01_class_distribution.png")
    )

    X_train_raw = train_df["Cleaned_Sentence"].fillna("")
    y_train = train_df["Emotion"]

    X_val_raw = val_df["Cleaned_Sentence"].fillna("")
    y_val = val_df["Emotion"]

    X_test_raw = test_df["Cleaned_Sentence"].fillna("")
    y_test = test_df["Emotion"]

    # 3. Chạy ma trận 12 thực nghiệm
    print("\n" + "=" * 70)
    print("🔬 CHẠY MA TRẬN 12 THỰC NGHIỆM (3 Representations x 4 Models)")
    print("=" * 70)

    vectorizers = get_all_vectorizers()
    results_records = []

    best_score = -1.0
    best_config = None
    best_pipeline = None

    for rep_name, vec in vectorizers.items():
        print(f"\n📦 Trích xuất đặc trưng: {rep_name}...")
        t0 = time.time()
        X_train_feat = vec.fit_transform(X_train_raw)
        X_val_feat = vec.transform(X_val_raw)
        feat_time = time.time() - t0
        print(f"   Kích thước không gian đặc trưng: {X_train_feat.shape[1]} chiều (xử lý trong {feat_time:.2f}s)")

        models = get_base_models()
        for model_name, model in models.items():
            print(f"  👉 Huấn luyện: {model_name}...")
            t_start = time.time()
            model.fit(X_train_feat, y_train)
            train_time = time.time() - t_start

            # Dự đoán trên Train và Validation
            train_preds = model.predict(X_train_feat)
            val_preds = model.predict(X_val_feat)

            # Đánh giá metrics
            train_eval = evaluate_predictions(y_train, train_preds, labels)
            val_eval = evaluate_predictions(y_val, val_preds, labels)

            # Phân tích Gap Overfitting
            gap_info = compute_generalization_gap(train_eval["macro_f1"], val_eval["macro_f1"])

            rec = {
                "Representation": rep_name,
                "Model": model_name,
                "Features": X_train_feat.shape[1],
                "Train_Time_s": round(train_time, 3),
                "Train_Macro_F1": round(train_eval["macro_f1"], 4),
                "Val_Accuracy": round(val_eval["accuracy"], 4),
                "Val_Balanced_Acc": round(val_eval["balanced_accuracy"], 4),
                "Val_Macro_P": round(val_eval["macro_precision"], 4),
                "Val_Macro_R": round(val_eval["macro_recall"], 4),
                "Val_Macro_F1": round(val_eval["macro_f1"], 4),
                "Val_Weighted_F1": round(val_eval["weighted_f1"], 4),
                "Train_Val_Gap": round(gap_info["gap"], 4),
                "Status": gap_info["status"]
            }
            results_records.append(rec)
            print(f"     Val Macro-F1: {rec['Val_Macro_F1']:.4f} | Val Acc: {rec['Val_Accuracy']:.4f} | Gap: {rec['Train_Val_Gap']:.4f} ({rec['Status']})")

            # Cập nhật mô hình tốt nhất
            if val_eval["macro_f1"] > best_score:
                best_score = val_eval["macro_f1"]
                best_config = (rep_name, model_name)
                best_pipeline = (vec, model)

    results_df = pd.DataFrame(results_records)
    results_csv_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    results_df.to_csv(results_csv_path, index=False, encoding="utf-8")
    print(f"\n💾 Đã lưu bảng tổng hợp 12 thực nghiệm vào: {results_csv_path}")

    # In bảng xếp hạng
    print("\n🏆 BẢNG XẾP HẠNG HIỆU NĂNG 12 CẤU HÌNH (sắp xếp theo Val Macro-F1):")
    ranked = results_df.sort_values(by="Val_Macro_F1", ascending=False)
    print(ranked[["Representation", "Model", "Val_Accuracy", "Val_Macro_F1", "Train_Val_Gap", "Status"]].to_string(index=False))

    # 4. Vẽ biểu đồ so sánh 12 thực nghiệm
    plot_experiment_comparison(results_df, os.path.join(FIGURES_DIR, "02_model_comparison.png"))

    # 5. Thử nghiệm chiến lược Đa lớp (Buổi 04: One-vs-Rest vs One-vs-One)
    print("\n" + "=" * 70)
    print("⚔️  THỰC NGHIỆM CHIẾN LƯỢC ĐA LỚP: One-vs-Rest (OvR) vs One-vs-One (OvO)")
    print("=" * 70)
    best_rep_name, best_model_name = best_config
    best_vec, best_clf = best_pipeline

    X_train_best = best_vec.transform(X_train_raw)
    X_val_best = best_vec.transform(X_val_raw)

    from sklearn.linear_model import LogisticRegression
    ovr_clf = wrap_multiclass_strategy(LogisticRegression(C=1.0, max_iter=1000, random_state=42), strategy="ovr")
    ovo_clf = wrap_multiclass_strategy(LogisticRegression(C=1.0, max_iter=1000, random_state=42), strategy="ovo")

    t0 = time.time()
    ovr_clf.fit(X_train_best, y_train)
    t_ovr = time.time() - t0
    ovr_val_pred = ovr_clf.predict(X_val_best)
    ovr_eval = evaluate_predictions(y_val, ovr_val_pred, labels)

    t0 = time.time()
    ovo_clf.fit(X_train_best, y_train)
    t_ovo = time.time() - t0
    ovo_val_pred = ovo_clf.predict(X_val_best)
    ovo_eval = evaluate_predictions(y_val, ovo_val_pred, labels)

    print(f"  - One-vs-Rest (7 models): Time = {t_ovr:.2f}s | Val Acc = {ovr_eval['accuracy']:.4f} | Val Macro-F1 = {ovr_eval['macro_f1']:.4f}")
    print(f"  - One-vs-One (21 models): Time = {t_ovo:.2f}s | Val Acc = {ovo_eval['accuracy']:.4f} | Val Macro-F1 = {ovo_eval['macro_f1']:.4f}")

    # 6. Đánh giá mô hình tốt nhất trên Holdout Test Set (Chốt chặn cuối cùng)
    print("\n" + "=" * 70)
    print(f"🎯 ĐÁNH GIÁ MÔ HÌNH TỐI ƯU NHẤT TRÊN TẬP TEST ĐỘC LẬP: {best_config}")
    print("=" * 70)
    X_test_best = best_vec.transform(X_test_raw)
    test_preds = best_clf.predict(X_test_best)
    test_eval = evaluate_predictions(y_test, test_preds, labels)

    print(f"  ✅ Test Accuracy:         {test_eval['accuracy']:.4f}")
    print(f"  ✅ Test Balanced Acc:     {test_eval['balanced_accuracy']:.4f}")
    print(f"  ✅ Test Macro-Precision:  {test_eval['macro_precision']:.4f}")
    print(f"  ✅ Test Macro-Recall:     {test_eval['macro_recall']:.4f}")
    print(f"  ⭐ Test Macro-F1:         {test_eval['macro_f1']:.4f}")
    print(f"  ✅ Test Weighted-F1:      {test_eval['weighted_f1']:.4f}")

    # 7. Xuất Confusion Matrix Heatmap trên tập Test
    plot_confusion_matrix(
        test_eval["confusion_matrix"],
        labels=labels,
        title=f"Confusion Matrix (Normalized) — {best_config[1]} ({best_config[0]})",
        save_path=os.path.join(FIGURES_DIR, "03_confusion_matrix_best_model.png")
    )

    # 8. Khảo sát Bias-Variance & Regularization Path (Buổi 05)
    print("\n📈 Đang vẽ đồ thị Regularization Path & Learning Curve...")
    from sklearn.svm import LinearSVC
    c_list = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    plot_regularization_path(
        LinearSVC,
        X_train_best,
        y_train,
        X_val_best,
        y_val,
        c_values=c_list,
        title="Linear SVM trên UIT-VSMEC",
        save_path=os.path.join(FIGURES_DIR, "04_regularization_tuning_curve.png")
    )

    # 9. Đường cong học tập (Learning Curve)
    plot_learning_curve_graph(
        best_clf,
        X_train_best,
        y_train,
        title=f"{best_config[1]} ({best_config[0]})",
        save_path=os.path.join(FIGURES_DIR, "05_learning_curve_best_model.png")
    )

    # 10. Xuất phân tích ca sai sót (Error Analysis)
    test_df["Predicted_Emotion"] = test_preds
    test_df["Is_Correct"] = (test_df["Emotion"] == test_df["Predicted_Emotion"])
    error_cases = test_df[~test_df["Is_Correct"]].copy()
    error_csv_path = os.path.join(RESULTS_DIR, "error_cases_analysis.csv")
    error_cases.to_csv(error_csv_path, index=False, encoding="utf-8")
    print(f"🔍 Đã lưu {len(error_cases)} ca dự đoán sai để phân tích nguyên nhân vào: {error_csv_path}")

    print("\n" + "=" * 70)
    print("🎉 HOÀN TẤT TOÀN BỘ QUY TRÌNH THỰC NGHIỆM!")
    print(f"👉 Toàn bộ biểu đồ lưu tại: {FIGURES_DIR}")
    print(f"👉 Bảng metrics lưu tại:   {results_csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

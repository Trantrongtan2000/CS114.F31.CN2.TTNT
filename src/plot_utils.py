# -*- coding: utf-8 -*-
"""
Module: plot_utils.py
Chức năng: Trực quan hóa kết quả thực nghiệm theo chuẩn báo cáo học thuật UIT:
  1. Biểu đồ nhiệt ma trận nhầm lẫn (Confusion Matrix Heatmap)
  2. Đường cong học tập (Learning Curve) phân tích hội tụ
  3. Đường cong điều chuẩn (Regularization Path - C tuning) khảo sát Overfitting
  4. Biểu đồ cột so sánh Macro-F1 giữa 12 thực nghiệm
"""

import os
from typing import List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve


# Cấu hình style vẽ đồ thị chuẩn publication
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_class_distribution(y_train: pd.Series, save_path: str) -> None:
    """Vẽ biểu đồ phân bố 7 lớp cảm xúc minh họa Class Imbalance."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    counts = y_train.value_counts()

    plt.figure(figsize=(9, 5))
    palette = sns.color_palette("mako", len(counts))
    ax = sns.barplot(x=counts.index, y=counts.values, palette=palette, edgecolor="#333333", linewidth=0.5)

    plt.title("Phân bố nhãn cảm xúc trên tập huấn luyện UIT-VSMEC", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Trạng thái cảm xúc", fontsize=11)
    plt.ylabel("Số lượng mẫu", fontsize=11)
    plt.xticks(rotation=20, ha="right")

    # Hiển thị số lượng và % trên đầu mỗi cột
    total = len(y_train)
    for p in ax.patches:
        height = p.get_height()
        pct = height / total * 100
        ax.annotate(f"{int(height)}\n({pct:.1f}%)",
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Đã lưu biểu đồ phân bố lớp: {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str,
    save_path: str
) -> None:
    """Vẽ biểu đồ nhiệt ma trận nhầm lẫn (Heatmap) thể hiện tỷ lệ % chuẩn hóa."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Chuẩn hóa theo từng hàng (tỷ lệ dự đoán đúng trên từng lớp thực tế)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Tỷ lệ dự đoán"}
    )
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Nhãn dự đoán (Predicted Label)", fontsize=11)
    plt.ylabel("Nhãn thực tế (True Label)", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Đã lưu biểu đồ Confusion Matrix: {save_path}")


def plot_learning_curve_graph(
    estimator: Any,
    X: Any,
    y: Any,
    title: str,
    save_path: str,
    cv: int = 5
) -> None:
    """Vẽ đường cong học tập (Learning Curve) khảo sát Overfitting / Underfitting (Buổi 05)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        train_sizes=np.linspace(0.2, 1.0, 5),
        scoring="f1_macro",
        n_jobs=-1,
        random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, "o-", color="#d9534f", label="Training Macro-F1", linewidth=2)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#d9534f")

    plt.plot(train_sizes, val_mean, "s-", color="#0275d8", label="Cross-Validation Macro-F1", linewidth=2)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#0275d8")

    plt.title(f"Đường cong học tập (Learning Curve) — {title}", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Số lượng mẫu huấn luyện (Training Examples)", fontsize=11)
    plt.ylabel("Macro-F1 Score", fontsize=11)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Đã lưu biểu đồ Learning Curve: {save_path}")


def plot_regularization_path(
    model_class: Any,
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    c_values: List[float],
    title: str,
    save_path: str
) -> None:
    """Khảo sát biến thiên siêu tham số điều chuẩn C ảnh hưởng đến Bias-Variance."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    from sklearn.metrics import f1_score

    train_scores = []
    val_scores = []

    for c in c_values:
        clf = model_class(C=c, max_iter=2000, random_state=42)
        clf.fit(X_train, y_train)

        tr_pred = clf.predict(X_train)
        va_pred = clf.predict(X_val)

        train_scores.append(f1_score(y_train, tr_pred, average="macro", zero_division=0))
        val_scores.append(f1_score(y_val, va_pred, average="macro", zero_division=0))

    plt.figure(figsize=(8, 5))
    plt.plot(c_values, train_scores, "o-", color="#e74c3c", label="Train Macro-F1", linewidth=2)
    plt.plot(c_values, val_scores, "s-", color="#2980b9", label="Validation Macro-F1", linewidth=2)

    # Đánh dấu vùng Underfitting và Overfitting
    best_idx = int(np.argmax(val_scores))
    best_c = c_values[best_idx]
    plt.axvline(best_c, color="#27ae60", linestyle="--", alpha=0.7, label=f"C tối ưu = {best_c}")

    plt.xscale("log")
    plt.title(f"Khảo sát siêu tham số điều chuẩn C (Regularization Path) — {title}", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Tham số điều chuẩn C (thang đo log)", fontsize=11)
    plt.ylabel("Macro-F1 Score", fontsize=11)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Đã lưu biểu đồ Regularization Path: {save_path}")


def plot_experiment_comparison(results_df: pd.DataFrame, save_path: str) -> None:
    """Vẽ biểu đồ cột so sánh Macro-F1 giữa 12 thực nghiệm."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(11, 5.5))
    palette = sns.color_palette("Set2", len(results_df["Representation"].unique()))

    ax = sns.barplot(
        data=results_df,
        x="Model",
        y="Val_Macro_F1",
        hue="Representation",
        palette=palette,
        edgecolor="#333333",
        linewidth=0.6
    )

    plt.title("So sánh Macro-F1 trên tập Validation giữa 12 cấu hình thực nghiệm", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Họ mô hình Machine Learning", fontsize=11)
    plt.ylabel("Macro-F1 Score (Validation)", fontsize=11)
    plt.legend(title="Phương pháp trích xuất đặc trưng", loc="lower right", frameon=True)
    plt.ylim(0, 0.75)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}",
                        xy=(p.get_x() + p.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Đã lưu biểu đồ so sánh thực nghiệm: {save_path}")

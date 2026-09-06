# -*- coding: utf-8 -*-
"""
Module: evaluate.py
Chức năng: Tính toán toàn diện các độ đo đánh giá mô hình học máy:
  - Accuracy & Balanced Accuracy
  - Macro-Precision, Macro-Recall, Macro-F1 (trọng tâm bài toán mất cân bằng lớp)
  - Weighted-F1
  - Confusion Matrix (ma trận nhầm lẫn 7x7)
  - Train vs Validation Gap (định lượng hiện tượng Overfitting)
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str]
) -> Dict[str, Any]:
    """
    Tính toán toàn bộ bộ chỉ số hiệu năng trên một tập dữ liệu.
    """
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm,
        "classification_report": report_dict
    }


def compute_generalization_gap(train_f1: float, val_f1: float) -> Dict[str, Any]:
    """
    Định lượng khoảng cách hiệu năng để phân tích Overfitting / Underfitting (Buổi 05):
      - Gap = Train_F1 - Val_F1
      - Gap > 0.15: Cảnh báo Overfitting nghiêm trọng
      - Gap < 0.05 và F1 thấp: Dấu hiệu Underfitting
    """
    gap = train_f1 - val_f1
    if gap > 0.15:
        status = "Overfitting"
    elif val_f1 < 0.45 and train_f1 < 0.50:
        status = "Underfitting"
    else:
        status = "Good Generalization"

    return {
        "train_f1": train_f1,
        "val_f1": val_f1,
        "gap": gap,
        "status": status
    }

# -*- coding: utf-8 -*-
"""
Module: models.py
Chức năng: Khởi tạo và cấu hình 4 họ mô hình học máy theo chuẩn đề cương CS114:
  1. Multinomial Naive Bayes (Probabilistic Baseline)
  2. Logistic Regression (Linear / Softmax)
  3. Linear Support Vector Machine (Maximum Margin)
  4. Random Forest (Ensemble Bagging)
Đồng thời hỗ trợ chiến lược phân loại đa lớp: One-vs-Rest (OvR) và One-vs-One (OvO).
"""

from typing import Dict, Any
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier


def get_base_models(random_state: int = 42) -> Dict[str, Any]:
    """Khởi tạo 4 mô hình học máy cơ bản với siêu tham số chuẩn."""
    return {
        "MultinomialNB": MultinomialNB(alpha=0.5),
        "LogisticRegression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs"
        ),
        "LinearSVM": LinearSVC(
            C=1.0,
            max_iter=2000,
            random_state=random_state,
            dual="auto"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=30,
            random_state=random_state,
            n_jobs=-1
        )
    }


def wrap_multiclass_strategy(base_estimator: Any, strategy: str = "ovr") -> Any:
    """
    Bọc mô hình bằng chiến lược phân loại đa lớp (Buổi 04):
      - 'ovr': One-vs-Rest (7 mô hình nhị phân cho 7 lớp)
      - 'ovo': One-vs-One (21 mô hình nhị phân cho 21 cặp lớp)
    """
    if strategy.lower() == "ovr":
        return OneVsRestClassifier(base_estimator, n_jobs=-1)
    elif strategy.lower() == "ovo":
        return OneVsOneClassifier(base_estimator, n_jobs=-1)
    else:
        raise ValueError(f"Chiến lược không hợp lệ: {strategy}. Chỉ nhận 'ovr' hoặc 'ovo'.")

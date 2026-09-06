# -*- coding: utf-8 -*-
"""
Module: feature_builder.py
Chức năng: Trích xuất đặc trưng văn bản theo 3 thiết kế thực nghiệm:
  - Experiment A: Word-level TF-IDF (1-gram, 2-gram)
  - Experiment B: Character n-gram TF-IDF (2..5 grams)
  - Experiment C: Hybrid FeatureUnion (kết hợp cả Word và Char TF-IDF)
"""

from typing import Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_word_vectorizer(
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 3,
    max_features: int = 3000
) -> TfidfVectorizer:
    """
    Experiment A: Word-level TF-IDF
    Bắt ngữ nghĩa từ đơn và từ ghép tiếng Việt (đã được nối bằng dấu _).
    """
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2"
    )


def build_char_vectorizer(
    ngram_range: Tuple[int, int] = (2, 5),
    min_df: int = 5,
    max_features: int = 5000
) -> TfidfVectorizer:
    """
    Experiment B: Character n-gram TF-IDF
    Bắt cấu trúc hình thái ký tự, biến thể chính tả, teencode, icon.
    """
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2"
    )


def build_hybrid_vectorizer(
    word_max_features: int = 2500,
    char_max_features: int = 3500
) -> FeatureUnion:
    """
    Experiment C: Hybrid FeatureUnion
    Kết hợp đồng thời cả biểu diễn cấp độ từ và cấp độ ký tự.
    """
    word_vec = build_word_vectorizer(max_features=word_max_features)
    char_vec = build_char_vectorizer(max_features=char_max_features)

    return FeatureUnion([
        ("word_tfidf", word_vec),
        ("char_tfidf", char_vec)
    ])


def get_all_vectorizers() -> Dict[str, Any]:
    """Trả về dictionary chứa 3 phương pháp trích xuất đặc trưng."""
    return {
        "Word_TFIDF": build_word_vectorizer(),
        "Char_TFIDF": build_char_vectorizer(),
        "Hybrid_TFIDF": build_hybrid_vectorizer()
    }

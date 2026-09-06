# -*- coding: utf-8 -*-
"""
Module: preprocessor.py
Chức năng: Tiền xử lý văn bản tiếng Việt từ mạng xã hội (UIT-VSMEC)
Bao gồm:
  - Chuẩn hóa Unicode (NFC)
  - Ánh xạ Emoji sang Token cảm xúc đặc trưng
  - Xử lý Teencode / Viết tắt phổ biến trên mạng xã hội
  - Rút gọn ký tự lặp cảm xúc (vd: "vuiiiii" -> "vui", "buồnnnn" -> "buồn")
  - Phân đoạn từ tiếng Việt (Word Segmentation) bằng PyVi / Underthesea
  - Giữ lại các từ phủ định quan trọng ("không", "chẳng", "chưa") khi lọc stop words
"""

import re
import unicodedata
from typing import List, Optional

# Thử import pyvi hoặc underthesea cho word segmentation
try:
    from pyvi import ViTokenizer
    HAS_PYVI = True
except ImportError:
    HAS_PYVI = False

try:
    import underthesea
    HAS_UNDERTHESEA = True
except ImportError:
    HAS_UNDERTHESEA = False


# 1. Từ điển ánh xạ Teencode & Viết tắt trên mạng xã hội
TEENCODE_DICT = {
    "ko": "không", "k": "không", "kh": "không", "hổng": "không", "hok": "không",
    "đc": "được", "dc": "được", "dk": "được", "đk": "được",
    "bt": "biết", "bit": "biết",
    "vs": "với", "w": "với",
    "thik": "thích", "thjk": "thích", "tik": "thích",
    "wa": "quá", "wá": "quá",
    "j": "gì", "z": "gì", "gi": "gì",
    "ntn": "như thế nào",
    "ms": "mới",
    "ng": "người", "ngta": "người ta", "mn": "mọi người",
    "nhiu": "nhiều", "nhìu": "nhiều",
    "bh": "bây giờ", "h": "giờ", "hnay": "hôm nay",
    "ch": "chưa", "chx": "chưa",
    "r": "rồi", "rùi": "rồi", "roy": "rồi",
    "mik": "mình", "mk": "mình",
    "t": "tao", "m": "mày",
    "ak": "à", "ah": "à", "uh": "ừ", "uhm": "ừm",
    "s": "sao", "sao z": "sao vậy", "sv": "sinh viên",
    "ib": "nhắn tin", "inbox": "nhắn tin",
    "cmt": "bình luận", "comment": "bình luận",
    "fb": "facebook",
    "vl": "rất", "vcl": "rất", "vkl": "rất", "vđ": "vấn đề",
    "đm": "chửi_thề", "đcm": "chửi_thề", "vcl": "chửi_thề", "dcm": "chửi_thề"
}

# 2. Ánh xạ Emoji & Ký tự cảm xúc phổ biến
EMOJI_DICT = {
    # Vui vẻ / Thích thú (Enjoyment)
    "😀": " icon_vui ", "😁": " icon_vui ", "😆": " icon_vui ", "😅": " icon_vui ",
    "😂": " icon_vui ", "🤣": " icon_vui ", "😊": " icon_vui ", "😇": " icon_vui ",
    "🙂": " icon_vui ", "😉": " icon_vui ", "😍": " icon_vui ", "🥰": " icon_vui ",
    "😘": " icon_vui ", "😋": " icon_vui ", "😝": " icon_vui ", "😎": " icon_vui ",
    ":))": " icon_vui ", ":)))": " icon_vui ", ":)": " icon_vui ", ":D": " icon_vui ",
    "^^": " icon_vui ", "^_^": " icon_vui ", "<3": " icon_vui ",
    # Buồn bã (Sadness)
    "😭": " icon_buon ", "😢": " icon_buon ", "😿": " icon_buon ", "😔": " icon_buon ",
    "😞": " icon_buon ", "😟": " icon_buon ", "🥺": " icon_buon ", "💔": " icon_buon ",
    ":(": " icon_buon ", ":((": " icon_buon ", ":(((": " icon_buon ", ":<": " icon_buon ",
    # Tức giận (Anger)
    "😡": " icon_tucgian ", "😠": " icon_tucgian ", "🤬": " icon_tucgian ",
    "👿": " icon_tucgian ", "💢": " icon_tucgian ",
    # Chán ghét (Disgust)
    "🤢": " icon_chankhet ", "🤮": " icon_chankhet ", "😒": " icon_chankhet ",
    "🙄": " icon_chankhet ", "😤": " icon_chankhet ",
    # Lo sợ (Fear)
    "😱": " icon_loso ", "😨": " icon_loso ", "😰": " icon_loso ",
    "😥": " icon_loso ", "😧": " icon_loso ", "😦": " icon_loso ",
    # Ngạc nhiên (Surprise)
    "😲": " icon_ngacnhien ", "😯": " icon_ngacnhien ", "🤯": " icon_ngacnhien ",
    "😳": " icon_ngacnhien ", "ô_kìa": " icon_ngacnhien "
}

# 3. Các từ phủ định BẮT BUỘC GIỮ LẠI
NEGATION_WORDS = {"không", "chẳng", "chưa", "chả", "đếch", "đéo", "đâu_có"}


def normalize_unicode(text: str) -> str:
    """Chuẩn hóa Unicode về chuẩn dựng sẵn (NFC)."""
    return unicodedata.normalize("NFC", text)


def normalize_repeated_chars(text: str) -> str:
    """Rút gọn ký tự lặp kéo dài: 'vuiiiii' -> 'vui', 'buồnnnn' -> 'buồn'."""
    # Lặp lại trên 2 lần thì chỉ giữ 1 lần
    return re.sub(r"([a-zA-Zà-ỹÀ-Ỹ])\1{2,}", r"\1", text)


def replace_teencode(text: str) -> str:
    """Thay thế các từ viết tắt / teencode bằng từ tiếng Việt chuẩn."""
    words = text.split()
    converted = [TEENCODE_DICT.get(w.lower(), w) for w in words]
    return " ".join(converted)


def map_emojis(text: str) -> str:
    """Ánh xạ emoji và emoticon thành token cảm xúc."""
    for em, token in EMOJI_DICT.items():
        text = text.replace(em, token)
    return text


def clean_special_chars(text: str) -> str:
    """Làm sạch ký tự thừa, giữ lại chữ cái tiếng Việt, số, và các icon cảm xúc."""
    # Giữ lại các token icon_...
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # Xóa URL
    text = re.sub(r"[\r\n\t]+", " ", text)             # Xóa ký tự xuống dòng
    # Giữ lại chữ, số, dấu gạch dưới (để nối từ), dấu cách và các token icon
    text = re.sub(r"[^\w\sà-ỹÀ-Ỹ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def segment_words(text: str, engine: str = "pyvi") -> str:
    """
    Phân đoạn từ tiếng Việt (Word Segmentation).
    Ví dụ: 'học sinh vui vẻ' -> 'học_sinh vui_vẻ'
    """
    if engine == "pyvi" and HAS_PYVI:
        return ViTokenizer.tokenize(text)
    elif HAS_UNDERTHESEA:
        return underthesea.word_tokenize(text, format="text")
    elif HAS_PYVI:
        return ViTokenizer.tokenize(text)
    else:
        # Fallback: không ghép từ
        return text


def preprocess_sentence(
    text: str,
    segment: bool = True,
    segment_engine: str = "pyvi"
) -> str:
    """
    Toàn bộ quy trình tiền xử lý 1 câu văn bản mạng xã hội:
    1. Chuẩn hóa Unicode NFC
    2. Ánh xạ Emoji sang Token cảm xúc
    3. Chuyển về chữ thường
    4. Thay thế Teencode / Viết tắt
    5. Rút gọn ký tự lặp
    6. Phân đoạn từ tiếng Việt (nối từ phức bằng _)
    7. Làm sạch ký tự đặc biệt
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Bước 1: Unicode NFC
    text = normalize_unicode(text)

    # Bước 2: Emoji & Emoticons
    text = map_emojis(text)

    # Bước 3: Lowercase
    text = text.lower()

    # Bước 4: Teencode
    text = replace_teencode(text)

    # Bước 5: Rút gọn lặp ký tự
    text = normalize_repeated_chars(text)

    # Bước 6: Phân đoạn từ (nối từ ghép để TF-IDF bắt được n-gram có nghĩa)
    if segment:
        text = segment_words(text, engine=segment_engine)

    # Bước 7: Làm sạch ký tự rác
    text = clean_special_chars(text)

    return text


def batch_preprocess(
    sentences: List[str],
    segment: bool = True,
    verbose: bool = True
) -> List[str]:
    """Tiền xử lý hàng loạt danh sách câu."""
    cleaned = []
    total = len(sentences)
    for i, s in enumerate(sentences):
        cleaned.append(preprocess_sentence(s, segment=segment))
        if verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{total} sentences...")
    return cleaned


if __name__ == "__main__":
    # Test mẫu
    sample = "lo học đi . yêu đương lol gì hay lại thích học sinh học :))) vuiiii quáaaaa 😡"
    print("Gốc:", sample)
    print("Xử lý:", preprocess_sentence(sample))

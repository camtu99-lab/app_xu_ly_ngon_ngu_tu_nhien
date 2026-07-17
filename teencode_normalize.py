# -*- coding: utf-8 -*-
"""
teencode_normalize.py
----------------------
Chuẩn hoá teencode / viết tắt tiếng Việt phổ biến trên mạng xã hội thành từ đầy đủ,
TRƯỚC khi đưa vào bước làm sạch văn bản (clean_text) và biểu diễn TF-IDF.

Lý do cần bước này:
    Mô hình được huấn luyện trên văn bản chuẩn (văn phong báo chí), nên khi người dùng
    nhập câu kiểu chat/mạng xã hội (vd: "ko", "vs", "dtich", "qtrong", "t.tiết"...),
    các từ khoá quan trọng bị "biến dạng" thành token lạ mà mô hình chưa từng thấy,
    khiến dự đoán sai hoặc có độ tin cậy thấp một cách không đáng có.

Cách hoạt động:
    1. Xoá dấu chấm dùng để viết tắt kiểu "t.tiết", "k.thức" (ghép liền chữ).
    2. Rút gọn ký tự lặp quá nhiều (vd: "đẹppp" -> "đẹp").
    3. Thay thế từng từ/cụm viết tắt bằng từ đầy đủ theo từ điển TEENCODE_MAP bên dưới.

Sinh viên có thể mở rộng thêm TEENCODE_MAP tuỳ theo loại văn bản thực tế cần xử lý.
"""

import re

# Từ điển teencode/viết tắt phổ biến -> từ/cụm từ đầy đủ tương ứng.
# (Khoá phải là chữ thường, không dấu chấm/câu, vì được áp dụng SAU bước xoá dấu chấm.)
TEENCODE_MAP = {
    # Đại từ nhân xưng / xưng hô viết tắt
    "mn": "mọi người", "ace": "anh chị em", "acE": "anh chị em",
    "tui": "tôi", "mik": "mình", "mk": "mình", "b": "bạn", "bn": "bạn", "ng": "người",
    "ngta": "người ta", "nta": "người ta",

    # Phủ định / khẳng định
    "ko": "không", "k": "không", "hok": "không", "hong": "không", "k0": "không",
    "hem": "không", "kg": "không",
    "ok": "đồng ý", "oki": "đồng ý", "okie": "đồng ý",

    # Từ nối / trợ từ hay viết tắt
    "vs": "với", "dc": "được", "đc": "được", "dk": "được", "r": "rồi", "roài": "rồi",
    "olz": "rồi", "j": "gì", "z": "vậy", "v": "vậy", "nhma": "nhưng mà", "nhg": "nhưng",
    "trc": "trước", "vt": "viết", "vc": "việc", "hnay": "hôm nay",

    # Viết tắt bằng cách bỏ dấu chấm câu (đã qua bước xoá dấu chấm ở trên, vẫn giữ dấu thanh)
    # vd: "t.tiết" -> "ttiết" sau khi xoá dấu chấm (không phải "ttiet")
    "ttiết": "thời tiết", "ttiet": "thời tiết",
    "qtrong": "quan trọng", "qtrọng": "quan trọng",
    "dtich": "diện tích", "dtích": "diện tích",
    "kthuc": "kiến thức", "kthức": "kiến thức",

    # Thuật ngữ vay mượn tiếng Anh phổ biến trong chat
    "training": "huấn luyện đào tạo", "trainning": "huấn luyện đào tạo",
    "info": "thông tin", "confirm": "xác nhận", "check": "kiểm tra",
    "fix": "sửa", "update": "cập nhật",

    # Khác
    "toang": "hỏng hoàn toàn", "cx": "cũng", "bik": "biết", "bít": "biết",
    "stt": "trạng thái", "add": "thêm", "sp": "sản phẩm", "ntn": "như thế nào",
}


def _collapse_repeated_chars(text: str) -> str:
    """"đẹppp quáaaa" -> "đẹp quá" (rút gọn ký tự lặp liên tiếp >=3 lần thành 1)."""
    return re.sub(r"(.)\1{2,}", r"\1", text)


def normalize_teencode(text: str) -> str:
    text = str(text).lower()

    # "n/nay" -> "ngày nay" (mẫu viết tắt có dấu gạch chéo)
    text = re.sub(r"\bn/nay\b", "ngày nay", text)

    # Xoá dấu chấm nằm giữa các chữ cái viết tắt kiểu "t.tiết", "k.thức"
    text = re.sub(r"(?<=[a-zàáảãạ])\.(?=[a-zàáảãạ])", "", text)

    text = _collapse_repeated_chars(text)

    _PUNCT = ".,!?;:\"'()[]{}…-"

    def lookup(tok: str) -> str:
        stripped = tok.strip(_PUNCT)
        if stripped in TEENCODE_MAP:
            # giữ lại dấu câu ở cuối (nếu có) để không phá cấu trúc câu
            prefix = tok[: len(tok) - len(tok.lstrip(_PUNCT))]
            suffix = tok[len(tok.rstrip(_PUNCT)):]
            return prefix + TEENCODE_MAP[stripped] + suffix
        return tok

    tokens = text.split()
    normalized_tokens = [lookup(tok) for tok in tokens]
    return " ".join(normalized_tokens)

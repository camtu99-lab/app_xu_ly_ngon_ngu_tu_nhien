# -*- coding: utf-8 -*-
"""
app.py — Ứng dụng demo Streamlit: PHÂN LOẠI CHỦ ĐỀ VĂN BẢN TIẾNG VIỆT (TF-IDF)

Chạy ứng dụng:
    streamlit run app.py

Yêu cầu thư viện (đã dùng trong project trước đó của bạn):
    pip install streamlit scikit-learn joblib pandas pymupdf

Chức năng:
    - Nhập trực tiếp một câu / đoạn văn bản  ->  dự đoán chủ đề đang được nói tới
    - Hoặc tải lên file PDF (dùng PyMuPDF/fitz để trích xuất văn bản) -> dự đoán chủ đề
    - Hiển thị top chủ đề khả năng cao nhất kèm % độ tin cậy (biểu đồ cột)
"""

import re
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from teencode_normalize import normalize_teencode

# ------------------------------------------------------------------
# Cấu hình trang
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Phân loại chủ đề văn bản tiếng Việt",
    page_icon="📚",
    layout="centered",
)

MODEL_DIR = "model"

# ------------------------------------------------------------------
# Tiền xử lý (PHẢI giống hệt bước tiền xử lý dùng khi huấn luyện mô hình)
# ------------------------------------------------------------------
VN_STOPWORDS = set("""
anh ai bao bà bị bởi bao_nhiêu cho các chị chúng chúng_ta cùng cũng còn của
do do_đó do_vậy dưới do đó đang đã đây đến để đó em gì giữa hay hoặc khá
khi lại lên là lúc mà mọi một mỗi nào nên nếu này nó nơi ngoài ngoài_ra
những nhau nhiều nếu ông qua ra rất sao sau sẽ tất tất_cả theo thì tôi
trên trong trước từ tại và vẫn về vào với xuống ta y_tá ấy đó
""".split())


def clean_text(text: str) -> str:
    text = normalize_teencode(text)  # Bước MỚI: chuẩn hoá teencode/viết tắt trước
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9À-ỹà-ỹ\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = [w for w in text.split() if w not in VN_STOPWORDS and len(w) > 1]
    return " ".join(tokens)


# ------------------------------------------------------------------
# Nạp mô hình (cache để không phải load lại mỗi lần tương tác)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Đang nạp mô hình...")
def load_model():
    vectorizer = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer.joblib")
    model = joblib.load(f"{MODEL_DIR}/best_topic_model.joblib")
    return vectorizer, model


def predict_topic(text: str, vectorizer, model, top_k: int = 5):
    cleaned = clean_text(text)
    if not cleaned.strip():
        return []
    vec = vectorizer.transform([cleaned])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
    else:
        scores = model.decision_function(vec)[0]
        exp_scores = np.exp(scores - np.max(scores))
        proba = exp_scores / exp_scores.sum()

    classes = model.classes_
    ranked = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def extract_text_from_pdf(uploaded_file) -> str:
    """Trích xuất toàn bộ văn bản từ file PDF bằng PyMuPDF (fitz)."""
    import fitz  # PyMuPDF
    file_bytes = uploaded_file.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text_parts = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(text_parts)


# ------------------------------------------------------------------
# Giao diện chính
# ------------------------------------------------------------------
st.title("📚 Phân loại chủ đề văn bản tiếng Việt")
st.markdown(
    "**Sinh viên thực hiện:** Vũ Thị Cẩm Tú &nbsp;&nbsp;|&nbsp;&nbsp; **MSHV:** CH_259463",
    unsafe_allow_html=True,
)
st.caption(
    "Nhập một câu / đoạn văn bản, hoặc tải lên file PDF — hệ thống sẽ dự đoán "
    "chủ đề mà câu/đoạn văn đó đang đề cập, dựa trên mô hình **TF-IDF** kết hợp "
    "thuật toán học máy đã huấn luyện."
)

try:
    vectorizer, model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error(
        "⚠️ Không tìm thấy mô hình đã huấn luyện trong thư mục `model/`.\n\n"
        "Hãy chạy notebook `notebook/Phan_loai_chu_de_TFIDF.ipynb` trước để sinh ra "
        "`tfidf_vectorizer.joblib` và `best_topic_model.joblib` trong thư mục `app/model/`."
    )

tab_text, tab_pdf = st.tabs(["✍️ Nhập văn bản", "📄 Tải file PDF"])

input_text = ""

with tab_text:
    input_text = st.text_area(
        "Nhập một câu hoặc đoạn văn bản:",
        height=160,
        placeholder="Ví dụ: Giá tôm nguyên liệu tại Đồng bằng sông Cửu Long tăng mạnh do nhu cầu xuất khẩu phục hồi...",
    )
    run_text_btn = st.button("🔍 Dự đoán chủ đề", type="primary", key="btn_text")

with tab_pdf:
    uploaded_pdf = st.file_uploader("Tải lên file PDF cần phân loại", type=["pdf"])
    run_pdf_btn = st.button("🔍 Dự đoán chủ đề", type="primary", key="btn_pdf")
    if uploaded_pdf is not None:
        with st.expander("Xem trước văn bản trích xuất từ PDF"):
            try:
                pdf_text_preview = extract_text_from_pdf(uploaded_pdf)
                uploaded_pdf.seek(0)
                st.text(pdf_text_preview[:2000] + ("..." if len(pdf_text_preview) > 2000 else ""))
            except Exception as e:
                st.warning(f"Không thể đọc file PDF: {e}")


def render_prediction(text_to_predict: str):
    if not model_loaded:
        return
    if not text_to_predict or not text_to_predict.strip():
        st.warning("Vui lòng nhập văn bản hoặc tải file PDF có nội dung.")
        return

    ranked = predict_topic(text_to_predict, vectorizer, model, top_k=8)
    if not ranked:
        st.warning("Không trích xuất được nội dung hợp lệ để phân loại.")
        return

    top_topic, top_score = ranked[0]
    st.success(f"### 🏷️ Chủ đề dự đoán: **{top_topic}**  ({top_score:.1%} độ tin cậy)")

    result_df = pd.DataFrame(ranked, columns=["Chủ đề", "Độ tin cậy"])
    result_df["Độ tin cậy (%)"] = (result_df["Độ tin cậy"] * 100).round(2)
    st.bar_chart(result_df.set_index("Chủ đề")["Độ tin cậy (%)"])
    st.dataframe(result_df[["Chủ đề", "Độ tin cậy (%)"]], hide_index=True, use_container_width=True)


if run_text_btn:
    render_prediction(input_text)

if run_pdf_btn:
    if uploaded_pdf is None:
        st.warning("Vui lòng chọn một file PDF trước khi bấm nút dự đoán.")
    else:
        with st.spinner("Đang trích xuất văn bản từ PDF..."):
            try:
                pdf_text = extract_text_from_pdf(uploaded_pdf)
            except Exception as e:
                pdf_text = ""
                st.error(f"Lỗi khi đọc file PDF: {e}")
        render_prediction(pdf_text)

st.divider()
with st.expander("ℹ️ Thông tin mô hình"):
    st.write(
        """
        - **Biểu diễn văn bản:** TF-IDF (unigram + bigram, tối đa 8000 đặc trưng)
        - **Mô hình phân loại:** được chọn tự động là mô hình có F1-macro cao nhất
          trong số 4 mô hình: Multinomial Naive Bayes, Logistic Regression,
          Linear SVM, Random Forest (xem chi tiết trong notebook huấn luyện).
        - **Số lớp chủ đề:** 8 (Nông nghiệp, Trồng trọt, Chăn nuôi, Thủy sản,
          Văn hóa, Giáo dục, Y tế, Du lịch), mỗi lớp ≥ 500 mẫu dữ liệu huấn luyện.
        """
    )

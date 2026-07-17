# Phân loại chủ đề văn bản tiếng Việt — Deploy lên Streamlit Community Cloud

## Cấu trúc thư mục (upload đúng như thế này lên GitHub)

```
(gốc repo)
├── app.py
├── teencode_normalize.py
├── requirements.txt
└── model/
    ├── tfidf_vectorizer.joblib
    ├── best_topic_model.joblib
    └── model_info.txt
```

⚠️ **Quan trọng:** `model/` phải nằm ngay cấp gốc của repo (cùng cấp với `app.py`), không được để trong thư mục con khác, vì code gọi đường dẫn tương đối `model/tfidf_vectorizer.joblib`.

## Các bước deploy

1. Tạo tài khoản GitHub (nếu chưa có): https://github.com/signup
2. Tạo repository mới (Public): https://github.com/new
3. Upload toàn bộ các file/thư mục ở trên vào repo (kéo thả qua giao diện web, hoặc dùng Git).
4. Vào https://share.streamlit.io/ → đăng nhập bằng GitHub → **Create app**.
5. Chọn đúng repo vừa tạo, branch `main`, main file path = `app.py` → **Deploy**.
6. Chờ 2–5 phút, nhận link cố định dạng `https://<tên-app>.streamlit.app`.

## Lưu ý

- App miễn phí sẽ "ngủ" nếu không có ai truy cập trong vài ngày — chỉ cần mở link, đợi ~30 giây để app khởi động lại.
- Nếu muốn cập nhật mô hình mới sau này: chỉ cần thay 2 file `.joblib` trong `model/` trên GitHub rồi app sẽ tự triển khai lại (redeploy) sau vài phút.

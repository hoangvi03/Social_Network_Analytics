# Dự Án Phân Tích và Dự Đoán Sự Quan Tâm Người Dùng Mạng Xã Hội Twitter

## Mô Tả Dự Án

Dự án này phát triển một hệ thống phân tích hành vi và dự đoán sự quan tâm của người dùng trên mạng xã hội Twitter. Hệ thống sử dụng các kỹ thuật machine learning và xử lý ngôn ngữ tự nhiên để:

- Phân tích hành vi người dùng dựa trên nội dung tweet
- Phân cụm người dùng theo sở thích và hành vi 
- Xác định chủ đề quan tâm thông qua mô hình LDA
- Dự đoán sự quan tâm của người dùng mới

## Cấu Trúc Dự Án

```
PTMXH_Detai5/
├── dataset_tweet.csv         # Dữ liệu tweet gốc (1.6M tweets)
├── train_model.ipynb         # Notebook huấn luyện mô hình chính
├── dudoanchude.ipynb         # Notebook dự đoán cho người dùng mới
├── tfidf_model.pkl          # Mô hình TF-IDF đã huấn luyện
├── kmeans_model.pkl         # Mô hình K-Means clustering
├── lda_model.pkl            # Mô hình LDA phân tích chủ đề
├── svd_model.pkl            # Mô hình SVD giảm chiều dữ liệu
├── dictionary.pkl           # Từ điển Gensim cho LDA
└── README.md                # Tài liệu hướng dẫn
```

## Yêu Cầu Hệ Thống

### Thư Viện Python
```
pandas
dask
scikit-learn
seaborn
matplotlib
gensim
nltk
pyarrow
joblib
numpy
```

### Cài Đặt
```bash
pip install pandas dask scikit-learn seaborn matplotlib gensim nltk pyarrow joblib numpy
```

## Quy Trình Thực Hiện

### 1. Tiền Xử Lý Dữ Liệu

**File:** `train_model.ipynb` (Cell 1-2)

- Đọc dữ liệu từ `dataset_tweet.csv` với 1.6M tweets
- Lấy mẫu 500,000 tweets do hạn chế tài nguyên
- Làm sạch dữ liệu: loại bỏ URL, ký tự đặc biệt, stopwords

```python
# Cấu trúc dữ liệu
columns = ["target", "ids", "date", "flag", "user", "text"]
# target: 0 (tiêu cực), 4 (tích cực)
```

### 2. Phân Tích Hành Vi Người Dùng

**File:** `train_model.ipynb` (Cell 3-4)

#### 2.1 Phân Loại Hành Vi
- Dựa trên nhãn "target" để phân loại tweet tiêu cực/tích cực
- Thống kê hành vi theo từng người dùng

#### 2.2 Phát Hiện Sở Thích
- Sử dụng **TF-IDF** để trích xuất từ khóa quan trọng
- Lọc người dùng hoạt động (≥5 tweets)
- Vector hóa nội dung tweet với 1000 đặc trưng

#### 2.3 Phân Cụm Người Dùng
- **K-Means clustering** với 7 cụm
- **SVD** giảm chiều từ 1000 xuống 50 đặc trưng
- Đánh giá chất lượng bằng **Silhouette Score**

### 3. Mô Hình Người Dùng

**File:** `train_model.ipynb` (Cell 4-5)

#### 3.1 Biểu Diễn Vector
- Tạo vector TF-IDF trung bình cho mỗi người dùng
- Tính độ tương tự Cosine giữa các người dùng

#### 3.2 Phân Tích Chủ Đề
- **LDA (Latent Dirichlet Allocation)** với 7 chủ đề
- Gán chủ đề nổi bật cho mỗi người dùng
- Tính phân phối xác suất chủ đề

### 4. Dự Đoán Sự Quan Tâm

**File:** `dudoanchude.ipynb`

Hệ thống dự đoán cho người dùng mới thông qua:

1. **Tiền xử lý**: Làm sạch tweet mới
2. **TF-IDF**: Chuyển đổi thành vector đặc trưng
3. **SVD**: Giảm chiều dữ liệu
4. **K-Means**: Phân loại vào cụm người dùng
5. **LDA**: Xác định chủ đề quan tâm chính

```python
# Ví dụ sử dụng
new_tweets = ["I love listening to music and going to concerts", "Work is so stressful today"]
result = predict_user_interest(new_tweets)
```

**Kết quả trả về:**
- `cluster`: Cụm người dùng (0-6)
- `main_topic`: Chủ đề quan tâm chính (0-6)
- `top_keywords`: Top 5 từ khóa đặc trưng

## Đánh Giá và Trực Quan Hóa

### Các Chỉ Số Đánh Giá
- **Silhouette Score**: Đánh giá chất lượng phân cụm
- **SSE (Sum of Squared Errors)**: Tối ưu số lượng cụm
- **Cosine Similarity**: Độ tương tự người dùng

### Trực Quan Hóa
- **Heatmap**: Ma trận độ tương tự người dùng
- **Bar Chart**: Phân bố chủ đề quan tâm
- **Time Series**: Xu hướng chủ đề theo thời gian
- **Elbow Method**: Tối ưu số cụm K-Means

## Kết Quả Đạt Được

1. **Phân cụm người dùng** thành 7 nhóm dựa trên hành vi và sở thích
2. **Xác định 7 chủ đề** quan tâm chính từ nội dung tweet
3. **Mô hình dự đoán** có thể phân loại người dùng mới
4. **Hệ thống gợi ý** dựa trên độ tương tự người dùng

## Hạn Chế và Hướng Phát Triển

### Hạn Chế
- Sử dụng mẫu giới hạn (500K/1.6M tweets) do tài nguyên
- Chỉ phân tích text tiếng Anh
- Chưa xử lý emoji và hashtag

### Hướng Phát Triển
- Tích hợp deep learning (BERT, Transformer)
- Phân tích đa ngôn ngữ
- Xử lý dữ liệu thời gian thực
- Gợi ý nội dung cá nhân hóa

## Tác Giả

Dự án thuộc Đề tài 5 - Chuyên đề Phân tích Mạng Xã Hội sinh viên:
- **Trần Hoàng Vĩ** - 2001216312
**Trường:** HUIT (Đại học Công nghiệp TP.HCM)

Dự án được phát triển cho mục đích học tập và nghiên cứu.

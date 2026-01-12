# Malware Multiclass Classification (Windows PE) — Colab-ready

Dự án này được thiết kế để:
- Đọc 4 feature sets (DLLs, APIs, PE Header, PE Section)
- Align theo `sha256` (intersection) để tránh mismatch
- Build `X` dạng **sparse CSR** (tối ưu RAM)
- Split **train/val/test stratified** (KHÔNG leakage)
- Train mô hình:
  - Logistic Regression (ElasticNet) — baseline mạnh cho sparse high-dim
  - LightGBM (tuỳ chọn) — thường cho accuracy cao hơn
- In/log rất nhiều thông tin để debug + tối ưu

## 1) Cấu trúc thư mục đề xuất

```
/content/drive/MyDrive/malware/
  DLLs_Imported.csv
  API_Functions.csv
  PE_Header.csv
  PE_Section.csv

  cache/                     # parquet cache tự tạo (tuỳ chọn)
  processed/                 # sparse dataset cache tự tạo (X_all.npz, y_all.npy, ...)
  outputs/
    logs/
    models/
    reports/
```

## 2) Chạy trên Google Colab

Trong Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

(Optional) cài LightGBM + Optuna:

```bash
pip -q install lightgbm optuna
```

Chạy train:

```bash
python run_colab.py
```

## 3) Anti-leakage

- Split dựa trên `y` (stratify) và index sample.
- Có check **sha256 overlap** giữa train/val/test → nếu overlap thì assert fail.
- Tuning chỉ dùng validation, test giữ nguyên đến cuối.

## 4) Output

- `outputs/models/model_<name>_<run_id>.joblib`
- `outputs/reports/metrics_<name>_<run_id>.json`
- `outputs/logs/train_<run_id>.log`
- Explainability:
  - Logistic: `top_features_logreg_<run_id>.json`
  - LightGBM: `feature_importance_lgbm_<run_id>.json`





Help:

1️⃣. Tạo môi trường ảo Python
---------------------------------------------------------------
py -3.12 -m venv venv
venv\Scripts\activate.bat       (Windows)
.\venv\Scripts\activate
source venv/bin/activate    (Linux/Mac)

2️⃣. Cài đặt thư viện cần thiết
---------------------------------------------------------------
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt wordnet omw-1.4
pip install pyvi
python -m nltk.downloader punkt punkt_tab
pip install google-generativeai
pip install python-dotenv
pip install matplotlib
pip install scikit-learn


pip install pandas pyarrow
pip install duckdb
---------------------------------------------------------------

👤 Thông tin sinh viên

Họ và tên: Phạm Ngọc Hưng

MSSV: 20235342

🏫 Trường: Đại học Bách khoa Hà Nội (HUST)

📘 Môn học: Project I – IT3150

👨‍🏫 Giảng viên hướng dẫn: Thầy Hoàng Việt Dũng

🛡️ Chủ đề: Nhận biết cơ bản về mã độc (malware)
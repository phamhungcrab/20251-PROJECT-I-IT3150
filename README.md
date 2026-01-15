# Phân loại đa lớp mã độc Windows PE (LightGBM)

Dự án này tập trung vào bài toán phân loại malware đa lớp từ đặc trưng tĩnh của file Windows PE. Pipeline ưu tiên tính tái lập, tránh leakage, và tối ưu hiệu năng trên dữ liệu sparse.

## Tổng quan

- 4 nhóm đặc trưng: DLLs, APIs, PE Header, PE Section.
- Align theo `sha256` để tránh lệch mẫu.
- Build ma trận đặc trưng dạng CSR (sparse) để giảm RAM.
- Chia train/val/test theo stratified; **không dùng test khi chọn tham số**.
- Mô hình:
  - Logistic Regression (tùy chọn).
  - LightGBM (mặc định, hiệu quả tốt).
- Explainability:
  - Feature importance tổng thể.
  - SHAP per-class (nếu cài `shap`).

## Nguồn dữ liệu

Kaggle dataset: **Windows Malwares** (Joakim Arvidsson), gồm 4 file CSV:
`DLLs_Imported.csv`, `API_Functions.csv`, `PE_Header.csv`, `PE_Section.csv`.  
License: **CC BY 4.0**.

## Cấu trúc thư mục dữ liệu

```
data_dir/
  DLLs_Imported.csv
  API_Functions.csv
  PE_Header.csv
  PE_Section.csv

  cache/        # (tùy chọn) cache parquet
  processed/    # cache sparse: X_all.npz, y_all.npy, sha256_all.npy, feature_names.json
  outputs/
    logs/
    models/
    reports/
```

> `data_dir` mặc định cấu hình trong `malware_multiclass_project/src/config.py`.  
> Trên Colab có thể là `/content/drive/MyDrive/malware`.

## Cài đặt

Tạo môi trường ảo (Windows):

```bash
py -3.12 -m venv venv
venv\Scripts\activate
```

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

Các gói cần thêm cho pipeline chính:

```bash
pip install lightgbm
```

(Tuỳ chọn) Explainability per-class:

```bash
pip install shap
```

## Chạy training

Local:

```bash
python -m malware_multiclass_project.src.train
```

Colab:

```bash
python malware_multiclass_project/run_colab.py
```

## Cấu hình nhanh

Sửa trong `malware_multiclass_project/src/config.py`:

- `data_dir`, `cache_dir`, `out_dir`
- `use_logreg` (False nếu chỉ dùng LightGBM)
- `lgbm_*` (tham số LightGBM)
- `lgbm_tune = False` (không dùng Optuna)
- `use_refine_pair = False` (mặc định tắt)
- `use_calibration = False` (mặc định tắt)

## Output

- `outputs/models/model_<name>_<run_id>.joblib`
- `outputs/reports/metrics_<name>_<run_id>.json`
- `outputs/logs/train_<run_id>.log`
- `outputs/reports/feature_importance_lgbm_<run_id>.json`
- `outputs/reports/shap_importance_lgbm_<run_id>.json` (nếu cài `shap`)

## Anti-leakage & tái lập

- Split theo stratified và kiểm tra trùng `sha256` giữa các tập.
- `random_state` cố định trong config để tái lập kết quả.

## Thông tin sinh viên

- Họ và tên: Phạm Ngọc Hồng
- MSSV: 20235342
- Trường: Đại học Bách khoa Hà Nội (HUST)
- Môn học: Project I – IT3150
- Giảng viên hướng dẫn: Thầy Hoàng Việt Dũng
- Chủ đề: Nhận biết cơ bản về mã độc (malware)

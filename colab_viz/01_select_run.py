import json
import re
from pathlib import Path

REPORTS_DIR = Path(BASE_DIR) / "outputs" / "reports"
MODELS_DIR = Path(BASE_DIR) / "outputs" / "models"

metrics_files = sorted(REPORTS_DIR.glob("metrics_*_*.json"))
if not metrics_files:
    raise FileNotFoundError(f"No metrics_*.json found in {REPORTS_DIR}")

latest = max(metrics_files, key=lambda p: p.stat().st_mtime)
match = re.match(r"metrics_(.+)_(\d{8}_\d{6})\.json", latest.name)
if not match:
    raise ValueError(f"Unexpected metrics filename: {latest.name}")

MODEL_NAME = match.group(1)
RUN_ID = match.group(2)

METRICS_PATH = latest
MODEL_PATH = MODELS_DIR / f"model_{MODEL_NAME}_{RUN_ID}.joblib"
FEAT_IMP_PATH = REPORTS_DIR / f"feature_importance_lgbm_{RUN_ID}.json"
SHAP_PATH = REPORTS_DIR / f"shap_importance_lgbm_{RUN_ID}.json"
CLASSES_PATH = REPORTS_DIR / f"classes_{RUN_ID}.json"

print(f"[OK] RUN_ID     = {RUN_ID}")
print(f"[OK] MODEL     = {MODEL_NAME}")
print(f"[OK] METRICS   = {METRICS_PATH}")
print(f"[OK] MODEL     = {MODEL_PATH}")
print(f"[OK] FEAT_IMP  = {FEAT_IMP_PATH}")
print(f"[OK] SHAP      = {SHAP_PATH}")
print(f"[OK] CLASSES   = {CLASSES_PATH}")

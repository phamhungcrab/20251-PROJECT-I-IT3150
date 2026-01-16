import json
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

if "MODEL_PATH" not in globals():
    raise RuntimeError("Run 01_select_run.py first to set MODEL_PATH.")

processed_dir = Path(BASE_DIR) / "processed"
X_path = processed_dir / "X_all.npz"
y_path = processed_dir / "y_all.npy"

if not X_path.exists() or not y_path.exists():
    raise FileNotFoundError("Processed data not found. Check processed/ folder.")

payload = joblib.load(MODEL_PATH)
model = payload["model"]
cfg = payload.get("config", {})

test_size = float(cfg.get("test_size", 0.2))
val_size = float(cfg.get("val_size", 0.2))
random_state = int(cfg.get("random_state", 42))

X = sp.load_npz(X_path)
y_str = np.load(y_path, allow_pickle=True).astype(str)

le = LabelEncoder()
y = le.fit_transform(y_str)
class_names = [str(c) for c in le.classes_]

if "CLASSES_PATH" in globals() and Path(CLASSES_PATH).exists():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    class_names = [
        mapping.get(str(i), mapping.get(i, class_names[i]))
        for i in range(len(class_names))
    ]

idx_all = np.arange(len(y))
idx_trainval, idx_test = train_test_split(
    idx_all,
    test_size=test_size,
    random_state=random_state,
    stratify=y,
)
val_rel = val_size / (1.0 - test_size)
_, idx_val = train_test_split(
    idx_trainval,
    test_size=val_rel,
    random_state=random_state,
    stratify=y[idx_trainval],
)

X_test = X[idx_test]
y_test = y[idx_test]

y_pred = model.predict(X_test)
p, r, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, labels=np.arange(len(class_names)), zero_division=0
)

df = pd.DataFrame(
    {
        "class": class_names,
        "precision": p,
        "recall": r,
        "f1": f1,
    }
)

df_melt = df.melt(id_vars="class", var_name="metric", value_name="value")
plt.figure(figsize=(12, 6))
sns.barplot(data=df_melt, x="class", y="value", hue="metric")
plt.xticks(rotation=45, ha="right")
plt.title(f"Per-class Precision/Recall/F1 (run {RUN_ID})")
plt.ylim(0.0, 1.0)
plt.tight_layout()
plt.show()

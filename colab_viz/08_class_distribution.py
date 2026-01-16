import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

processed_dir = Path(BASE_DIR) / "processed"
y_path = processed_dir / "y_all.npy"

if not y_path.exists():
    raise FileNotFoundError("Processed y_all.npy not found. Check processed/ folder.")

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

counts = np.bincount(y, minlength=len(class_names))
df = pd.DataFrame({"class": class_names, "count": counts})
df["ratio"] = df["count"] / df["count"].sum()

display(df)

plt.figure(figsize=(10, 4))
sns.barplot(data=df, x="class", y="count", palette="cubehelix")
plt.xticks(rotation=45, ha="right")
plt.title("Class Distribution")
plt.tight_layout()
plt.show()

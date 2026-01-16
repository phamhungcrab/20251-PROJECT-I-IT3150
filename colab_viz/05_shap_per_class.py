import json
import math
from pathlib import Path

if "SHAP_PATH" not in globals():
    raise RuntimeError("Run 01_select_run.py first to set SHAP_PATH.")

if not Path(SHAP_PATH).exists():
    raise FileNotFoundError(f"SHAP file not found: {SHAP_PATH}")

with open(SHAP_PATH, "r", encoding="utf-8") as f:
    shap_data = json.load(f)

classes = list(shap_data.keys())
topk = 15

cols = 2
rows = int(math.ceil(len(classes) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4))
axes = np.atleast_1d(axes).ravel()

for ax, cname in zip(axes, classes):
    items = shap_data[cname][:topk]
    df = pd.DataFrame(items).iloc[::-1]
    sns.barplot(data=df, x="mean_shap", y="feature", ax=ax, palette="magma")
    ax.set_title(f"Class: {cname}")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_ylabel("")

for ax in axes[len(classes):]:
    ax.axis("off")

plt.tight_layout()
plt.show()

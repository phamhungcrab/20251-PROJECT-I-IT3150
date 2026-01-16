import json
from pathlib import Path

from IPython.display import display

if "METRICS_PATH" not in globals():
    raise RuntimeError("Run 01_select_run.py first to set METRICS_PATH.")

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metrics = json.load(f)

rows = []
for split in ("val", "test", "val_refined", "test_refined", "val_calibrated", "test_calibrated"):
    if split not in metrics:
        continue
    for k, v in metrics[split].items():
        if isinstance(v, (int, float)):
            rows.append({"split": split, "metric": k, "value": float(v)})

if not rows:
    raise ValueError("No numeric metrics found in metrics JSON.")

df = pd.DataFrame(rows)
display(df.pivot_table(index="metric", columns="split", values="value"))

plt.figure(figsize=(10, 5))
sns.barplot(data=df, x="metric", y="value", hue="split")
plt.xticks(rotation=45, ha="right")
plt.title(f"Metrics by split (run {RUN_ID})")
plt.tight_layout()
plt.show()

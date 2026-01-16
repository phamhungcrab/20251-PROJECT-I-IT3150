import json
from pathlib import Path

if "FEAT_IMP_PATH" not in globals():
    raise RuntimeError("Run 01_select_run.py first to set FEAT_IMP_PATH.")

if not Path(FEAT_IMP_PATH).exists():
    raise FileNotFoundError(f"Feature importance file not found: {FEAT_IMP_PATH}")

with open(FEAT_IMP_PATH, "r", encoding="utf-8") as f:
    rows = json.load(f)

topk = 25
df = pd.DataFrame(rows).head(topk).iloc[::-1]

plt.figure(figsize=(10, 8))
sns.barplot(data=df, x="importance", y="feature", palette="viridis")
plt.title(f"Top {topk} Feature Importance (run {RUN_ID})")
plt.xlabel("Importance (split count)")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

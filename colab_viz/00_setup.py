import os
import sys
from pathlib import Path

BASE_DIR = os.environ.get("MALWARE_BASE_DIR", "/content/drive/MyDrive/malware")

try:
    from google.colab import drive

    drive.mount("/content/drive")
except Exception as exc:
    print(f"[WARN] drive.mount skipped: {exc}")

base_path = Path(BASE_DIR)
if not base_path.exists():
    raise FileNotFoundError(f"BASE_DIR not found: {BASE_DIR}")

src_candidates = [
    base_path / "src",
    base_path / "malware_multiclass_project" / "src",
    base_path / "malware_multiclass_project",
]
for p in src_candidates:
    if p.exists():
        sys.path.append(str(p))
        break

import importlib.util
import subprocess

def _pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pkg])

if importlib.util.find_spec("seaborn") is None:
    _pip_install("seaborn")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

print(f"[OK] BASE_DIR = {BASE_DIR}")

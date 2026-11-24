"""Which partition (train / val / test) holds each of the six trials without video?
Run in the same Kaggle notebook as verify_data.py (same inputs). Takes seconds."""
import json
from pathlib import Path

FUS = next(p.parent.parent for p in Path("/kaggle/input").rglob("meta.json")
           if "fusion" in str(p) and p.parent.name == "s01")
missing = {"s03": [40], "s05": [40], "s11": [38, 39, 40], "s14": [40]}   # 1-based trial numbers
for s, trials in missing.items():
    m = json.load(open(FUS / s / "meta.json"))
    for t in trials:
        i = t - 1                                   # meta.json uses 0-based trial indices
        part = ("train" if i in m["train_trials"] else
                "val" if i in m["val_trials"] else
                "TEST" if i in m["test_trials"] else "not found")
        print(f"{s} trial_{t:02d}: {part}")

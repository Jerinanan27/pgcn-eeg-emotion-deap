"""
kaggle_notebooks.py
===================
Copy each CELL string into a Kaggle notebook code cell.

Upload all 7 .py files as Kaggle Dataset: "pgcn-eeg-code" (PUBLIC)

ACCOUNT 1: jerinanan/data-preprocessed-python  +  pgcn-eeg-code
ACCOUNT 2: jerinanan/deap-fused-data           +  pgcn-eeg-code
ACCOUNT 3: pgcn-deap-fusion-preprocessed       +  pgcn-eeg-code
ACCOUNT 4: pgcn-deap-preprocessed              +  pgcn-eeg-code
"""

# ═══════════════════════════════════════════════════════════════════
# CELL 1 — Install PyG  (ALL accounts, run once per session)
# ═══════════════════════════════════════════════════════════════════
CELL_1 = """
import subprocess, sys, torch

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.version.cuda}")
print(f"GPU     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")

tv  = torch.__version__.split("+")[0]
cu  = "cu" + torch.version.cuda.replace(".", "")
url = f"https://data.pyg.org/whl/torch-{tv}+{cu}.html"

subprocess.run([sys.executable,"-m","pip","install",
    "pyg_lib","torch_scatter","torch_sparse",
    "torch_cluster","torch_spline_conv","-f",url,"-q"], check=False)
subprocess.run([sys.executable,"-m","pip","install","torch_geometric","-q"])
print("\\n✔ PyG installed")
"""

# ═══════════════════════════════════════════════════════════════════
# CELL 2 — Copy code files  (ALL accounts)
# ═══════════════════════════════════════════════════════════════════
CELL_2 = """
import shutil, os

for f in os.listdir("/kaggle/input/pgcn-eeg-code"):
    if f.endswith(".py"):
        shutil.copy(f"/kaggle/input/pgcn-eeg-code/{f}", f"/kaggle/working/{f}")
        print(f"  Copied: {f}")
print("✔ Code ready")
"""

# ═══════════════════════════════════════════════════════════════════
# CELL 3 — Verify data paths  (run and check output before continuing)
# ═══════════════════════════════════════════════════════════════════
CELL_3 = """
import torch, os, h5py, pickle, json

print(f"GPU : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── EEG .dat files (Account 1) ──
DAT = "/kaggle/input/datasets/jerinanan/data-preprocessed-python/data_preprocessed_python"
if os.path.exists(DAT):
    dats = sorted([f for f in os.listdir(DAT) if f.endswith(".dat")])
    print(f"\\n✔ DAT files : {len(dats)}  → {dats[:3]}")
    raw = pickle.load(open(f"{DAT}/s01.dat","rb"), encoding="latin1")
    print(f"  data shape  : {raw['data'].shape}")     # (40, 40, 8064)
    print(f"  labels shape: {raw['labels'].shape}")   # (40, 4)
else:
    print(f"\\n⚠ DAT not found: {DAT}")

# ── Fused H5 (Account 2) ──
H5 = "/kaggle/input/datasets/jerinanan/deap-fused-data/fused_dataset.h5"
if os.path.exists(H5):
    with h5py.File(H5,"r") as f:
        subs=list(f.keys()); s0,t0=subs[0],list(f[subs[0]].keys())[0]
        print(f"\\n✔ H5 : {len(subs)} subjects")
        print(f"  EEG   : {f[s0][t0]['raw_eeg'].shape}")
        print(f"  Video : {f[s0][t0]['video_embedding'].shape}")
else:
    print(f"\\n⚠ H5 not found: {H5}")

# ── Preprocessed datasets (Accounts 3 & 4) ──
for name,path in [("pgcn-deap-preprocessed",        "/kaggle/input/pgcn-deap-preprocessed"),
                   ("pgcn-deap-fusion-preprocessed", "/kaggle/input/pgcn-deap-fusion-preprocessed")]:
    if os.path.exists(path):
        subs=[d for d in os.listdir(path) if d.startswith("s")]
        mp=f"{path}/{sorted(subs)[0]}/meta.json"
        m=json.load(open(mp)) if os.path.exists(mp) else {}
        print(f"\\n✔ {name}: {len(subs)} subjects")
        print(f"  has_real_video={m.get('has_real_video','?')}")
        print(f"  class_weights keys: {list(m.get('class_weights',{}).keys())}")
    else:
        print(f"\\n⚠ Not mounted: {name}")
"""

# ═══════════════════════════════════════════════════════════════════
# CELL 4 — Smoke test  (ALL accounts — always run first, ~3 min)
#           Catches import/path errors before wasting GPU hours.
# ═══════════════════════════════════════════════════════════════════
CELL_4 = """
import sys, types
sys.path.insert(0, "/kaggle/working")

import pgcn_train as T
import pgcn_core  as C

# ── Patch for 2 subjects, 3 epochs ──
orig_init = C.DEAPDataset.__init__
def fast_init(self, root, subjects, split="all", no_hjorth=False):
    orig_init(self, root, subjects[:2], split, no_hjorth)
C.DEAPDataset.__init__ = fast_init

orig_cfgs = {k:dict(v) for k,v in T.CFGS.items()}
for k in T.CFGS: T.CFGS[k].update({"ep":3,"pat":2,"bs":16})

# ← change DATA to match your account's preprocessed folder
DATA = "/kaggle/input/pgcn-deap-preprocessed"          # Accounts 1, 4
# DATA = "/kaggle/input/pgcn-deap-fusion-preprocessed"  # Accounts 2, 3

args = types.SimpleNamespace(
    data_dir=DATA, out_dir="/kaggle/working/smoke",
    mode="within", fusion="none", ablation="none",
    device="cuda", workers=2, force=True,
)
T.run_within(args, T.ResultsStore())

# ── Restore ──
C.DEAPDataset.__init__ = orig_init
for k,v in orig_cfgs.items(): T.CFGS[k].update(v)
print("\\n✔ Smoke test PASSED — safe to run full experiments!")
"""

# ═══════════════════════════════════════════════════════════════════
# ACCOUNT 1 — CELL 5
# Datasets: jerinanan/data-preprocessed-python  +  pgcn-eeg-code
#
# After preprocessing finishes:
#   Output panel → right-click "preprocessed_eeg" → Save Version as Dataset
#   Name: pgcn-deap-preprocessed   Visibility: PUBLIC
# ═══════════════════════════════════════════════════════════════════
ACCOUNT_1_CELL5 = """
import os, sys, types
sys.path.insert(0, "/kaggle/working")

DAT = "/kaggle/input/datasets/jerinanan/data-preprocessed-python/data_preprocessed_python"
PRE = "/kaggle/working/preprocessed_eeg"
RES = "/kaggle/working/results_eeg"

# ── Step 1: Preprocess ──
# TIP: Run this on a CPU-only notebook first to save GPU quota
print("STEP 1: Preprocessing EEG .dat files ...")
ret = os.system(f"python /kaggle/working/preprocess_eeg.py --dat_dir {DAT} --out_dir {PRE}")
print(f"✔ ret={ret}")
# → Save output as Dataset: pgcn-deap-preprocessed (PUBLIC)

# ── Step 2: EEG-Only Within-Subject ──
import pgcn_train as T
from pgcn_core import ResultsStore

print("\\nSTEP 2: EEG-Only Within-Subject (32 subjects) ...")
args = types.SimpleNamespace(
    data_dir=PRE, out_dir=RES,
    mode="within", fusion="none", ablation="none",
    device="cuda", workers=2, force=False,
)
s1 = ResultsStore()
T.run_within(args, s1)
s1.summary(out_dir=f"{RES}/within_none/aggregate")
print("✔ EEG-Only Within done!")

# ── Step 3: EEG-Only LOSO ──
print("\\nSTEP 3: EEG-Only LOSO (32 folds) ...")
args.mode = "loso"
s2 = ResultsStore()
T.run_loso(args, s2)
s2.summary(out_dir=f"{RES}/loso_none/aggregate")
print("✔ EEG-Only LOSO done!")
"""

# ═══════════════════════════════════════════════════════════════════
# ACCOUNT 2 — CELL 5
# Datasets: jerinanan/deap-fused-data  +  pgcn-eeg-code
#
# After preprocessing finishes:
#   Save "preprocessed_fusion" as Dataset: pgcn-deap-fusion-preprocessed (PUBLIC)
# ═══════════════════════════════════════════════════════════════════
ACCOUNT_2_CELL5 = """
import os, sys, types
sys.path.insert(0, "/kaggle/working")

H5  = "/kaggle/input/datasets/jerinanan/deap-fused-data/fused_dataset.h5"
PRE = "/kaggle/working/preprocessed_fusion"
RES = "/kaggle/working/results_fusion"

# ── Step 1: Preprocess fusion H5 ──
print("STEP 1: Preprocessing fusion dataset (auto-detects real vs zero video) ...")
ret = os.system(f"python /kaggle/working/preprocess_fusion.py --h5_path {H5} --out_dir {PRE}")
print(f"✔ ret={ret}")
# Prints which subjects have real facial video
# → Save output as Dataset: pgcn-deap-fusion-preprocessed (PUBLIC)

# ── Step 2: Early Fusion Within-Subject (auto-filters to real-video subjects) ──
import pgcn_train as T
from pgcn_core import ResultsStore

print("\\nSTEP 2: Early Fusion Within-Subject ...")
args = types.SimpleNamespace(
    data_dir=PRE, out_dir=RES,
    mode="within", fusion="early", ablation="none",
    device="cuda", workers=2, force=False,
)
s1 = ResultsStore()
T.run_within(args, s1)
s1.summary(out_dir=f"{RES}/within_early/aggregate")
print("✔ Early Fusion Within done!")

# ── Step 3: Early Fusion LOSO ──
print("\\nSTEP 3: Early Fusion LOSO ...")
args.mode = "loso"
s2 = ResultsStore()
T.run_loso(args, s2)
s2.summary(out_dir=f"{RES}/loso_early/aggregate")
print("✔ Early Fusion LOSO done!")
"""

# ═══════════════════════════════════════════════════════════════════
# ACCOUNT 3 — CELL 5
# Datasets: pgcn-deap-fusion-preprocessed  +  pgcn-eeg-code
# ═══════════════════════════════════════════════════════════════════
ACCOUNT_3_CELL5 = """
import sys, types
sys.path.insert(0, "/kaggle/working")

PRE = "/kaggle/input/pgcn-deap-fusion-preprocessed"
RES = "/kaggle/working/results_fusion"

import pgcn_train as T
from pgcn_core import ResultsStore

# ── Late Fusion Within-Subject ──
print("Late Fusion Within-Subject ...")
args = types.SimpleNamespace(
    data_dir=PRE, out_dir=RES,
    mode="within", fusion="late", ablation="none",
    device="cuda", workers=2, force=False,
)
s1 = ResultsStore()
T.run_within(args, s1)
s1.summary(out_dir=f"{RES}/within_late/aggregate")
print("✔ Late Fusion Within done!")

# ── Late Fusion LOSO ──
print("\\nLate Fusion LOSO ...")
args.mode = "loso"
s2 = ResultsStore()
T.run_loso(args, s2)
s2.summary(out_dir=f"{RES}/loso_late/aggregate")
print("✔ Late Fusion LOSO done!")
"""

# ═══════════════════════════════════════════════════════════════════
# ACCOUNT 4 — CELL 5
# Datasets: pgcn-deap-preprocessed  +  pgcn-eeg-code
# ═══════════════════════════════════════════════════════════════════
ACCOUNT_4_CELL5 = """
import sys, types
sys.path.insert(0, "/kaggle/working")

PRE = "/kaggle/input/pgcn-deap-preprocessed"
RES = "/kaggle/working/results_eeg"

import pgcn_train as T
from pgcn_core import ResultsStore

ABLATIONS = [
    ("flat_gcn",     "A1 — Flat GCN (no hierarchy)"),
    ("no_band_attn", "A2 — No Band Attention"),
    ("struct_only",  "A3 — Structural Edges Only"),
    ("no_hjorth",    "A4 — No Hjorth Features"),
]

for abl_key, abl_name in ABLATIONS:
    print(f"\\n{'='*55}\\n  {abl_name}\\n{'='*55}")

    # Within-Subject
    args = types.SimpleNamespace(
        data_dir=PRE, out_dir=RES,
        mode="within", fusion="none", ablation=abl_key,
        device="cuda", workers=2, force=False,
    )
    s1 = ResultsStore()
    T.run_within(args, s1)
    s1.summary(out_dir=f"{RES}/within_none_abl_{abl_key}/aggregate")

    # LOSO
    args.mode = "loso"
    s2 = ResultsStore()
    T.run_loso(args, s2)
    s2.summary(out_dir=f"{RES}/loso_none_abl_{abl_key}/aggregate")
    print(f"✔ {abl_name} complete!")

print("\\n✔ ALL ABLATIONS DONE!")
"""

# ═══════════════════════════════════════════════════════════════════
# FINAL — Statistics + Plots  (run in any account after collecting all results)
# Upload merged results as Dataset: pgcn-all-results
# ═══════════════════════════════════════════════════════════════════
FINAL_CELL = """
import os, sys
sys.path.insert(0, "/kaggle/working")

os.system(
    "python /kaggle/working/pgcn_statistics_plots.py "
    "--results_eeg    /kaggle/input/pgcn-all-results/results_eeg "
    "--results_fusion /kaggle/input/pgcn-all-results/results_fusion "
    "--out_dir        /kaggle/working/figures"
)
print("✔ Figures saved to /kaggle/working/figures")
"""

if __name__ == "__main__":
    print("Notebook cell templates. Copy string contents into Kaggle code cells.")
    print("\nCELL_1          → Install PyG (all accounts)")
    print("CELL_2          → Copy code files (all accounts)")
    print("CELL_3          → Verify paths (all accounts)")
    print("CELL_4          → Smoke test (all accounts, always run first)")
    print("ACCOUNT_1_CELL5 → Preprocess EEG + EEG-Only Within + LOSO")
    print("ACCOUNT_2_CELL5 → Preprocess Fusion + Early Fusion Within + LOSO")
    print("ACCOUNT_3_CELL5 → Late Fusion Within + LOSO")
    print("ACCOUNT_4_CELL5 → All 4 Ablations Within + LOSO")
    print("FINAL_CELL      → Statistics + Plots")

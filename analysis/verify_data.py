"""
verify_data.py
==============
Checks on the SAVED datasets that the manuscript's Methods depend on.
CPU only, about 5 to 10 minutes. Paste into one Kaggle cell (or run as a script)
with these inputs attached: pgcn-deap-preprocessed, pgcn-deap-fusion-preprocessed,
deap-fused-data (fused_dataset.h5) and the DEAP data_preprocessed_python (.dat).

  1. Feature index 15 (approximate entropy) is zero in every saved window.
  2. Functional edges stored in the windows equal |Pearson| top-12 recomputed
     from the DEAP .dat EEG (shows which graph code produced the data).
  3. The EEG in fused_dataset.h5 equals the DEAP preprocessed .dat EEG
     (so the fusion data also passed DEAP's 4-45 Hz filter).
  4. EEG features and splits of the EEG-only and fusion datasets are identical
     for s01-s22.
  5. Number of trials with an all-zero video embedding for s01-s22.
  6. Power in 45-60 Hz relative to 30-45 Hz in the .dat EEG (filter roll-off).
Prints a short report and writes verify_data_report.txt.
"""
import json, pickle
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import signal

IN = Path("/kaggle/input")


def find_data(keyword):
    for p in IN.rglob("meta.json"):
        if keyword in str(p) and p.parent.name == "s01":
            return p.parent.parent
    raise FileNotFoundError(keyword)


EEG = find_data("pgcn-deap-preprocessed")
FUS = find_data("fusion")
DAT = next(IN.rglob("s01.dat")).parent
H5 = next(IN.rglob("fused_dataset.h5"))
FS, BASE, WIN, STEP = 128, 3 * 128, 4 * 128, 128
out = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def win(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def topk_abs_pearson(w, k=12):
    z = (w - w.mean(1, keepdims=True)) / (w.std(1, keepdims=True) + 1e-8)
    c = np.abs(z @ z.T / w.shape[1]); np.fill_diagonal(c, 0)
    return {(i, int(j)) for i in range(32) for j in np.argsort(c[i])[::-1][:k] if c[i, j] > 0}


say("EEG:", EEG, "\nFUS:", FUS, "\nDAT:", DAT, "\nH5 :", H5)

# 1. index 15
nz = 0; n = 0
for root in (EEG, FUS):
    for s in [f"s{i:02d}" for i in range(1, 33)]:
        for p in sorted((root / s).glob("trial_*/win_*.pt"))[::7]:     # every 7th window
            x = win(p).x; n += 1; nz += int(torch.any(x[:, 15] != 0))
say(f"1. windows checked: {n}; windows with non-zero feature 15: {nz}")

# 2. functional edges vs |Pearson| top-12
mism = 0; checked = 0
for s in ["s01", "s10", "s25"]:
    raw = pickle.load(open(DAT / f"{s}.dat", "rb"), encoding="latin1")["data"][:, :32, :]
    for t in [0, 17, 39]:
        for w_i in [0, 14, 28]:
            seg = raw[t][:, BASE + w_i * STEP: BASE + w_i * STEP + WIN]
            g = win(EEG / s / f"trial_{t:02d}" / f"win_{w_i:02d}.pt")
            stored = set(map(tuple, g.edge_index_func.t().tolist()))
            checked += 1; mism += int(stored != topk_abs_pearson(seg))
say(f"2. functional graphs checked: {checked}; differing from |Pearson| top-12: {mism}")

# 3. H5 EEG vs .dat EEG
with h5py.File(H5, "r") as f:
    diffs = []
    for s in ["s01", "s15", "s30"]:
        raw = pickle.load(open(DAT / f"{s}.dat", "rb"), encoding="latin1")["data"][:, :32, :]
        trials = sorted(f[s].keys())
        for t in [0, 20, 39]:
            h = np.array(f[s][trials[t]]["raw_eeg"], dtype=np.float32)[:32]
            diffs.append(float(np.abs(h[:, :raw.shape[2]] - raw[t]).max()) if h.shape == raw[t].shape else np.inf)
            if h.shape != raw[t].shape:
                say(f"   shape differs {s} trial {t}: h5 {h.shape} vs dat {raw[t].shape}")
    say(f"3. max |H5 raw_eeg - .dat EEG| over 9 trials: {max(diffs):.3g}")

    # 5. zero video embeddings
    zero = {}
    for s in [f"s{i:02d}" for i in range(1, 23)]:
        zero[s] = [t for t in sorted(f[s].keys())
                   if np.allclose(np.array(f[s][t]["video_embedding"]), 0, atol=1e-6)]
    tot = sum(len(v) for v in zero.values())
    say(f"5. s01-s22 trials with all-zero video embedding: {tot}")
    for s, v in zero.items():
        if v:
            say(f"   {s}: {len(v)} trials -> {v}")

# 4. EEG-only vs fusion datasets, s01-s22
maxd = 0.0; same_split = True
for s in [f"s{i:02d}" for i in range(1, 23)]:
    me, mf = json.load(open(EEG / s / "meta.json")), json.load(open(FUS / s / "meta.json"))
    same_split &= me["test_trials"] == mf["test_trials"] and me["train_trials"] == mf["train_trials"]
    for t in [0, 13, 27, 39]:
        a, b = win(EEG / s / f"trial_{t:02d}" / "win_05.pt"), win(FUS / s / f"trial_{t:02d}" / "win_05.pt")
        maxd = max(maxd, float((a.x - b.x).abs().max()))
say(f"4. identical splits: {same_split}; max |x_EEG - x_FUS| (88 windows): {maxd:.3g}")

# 6. 45-60 Hz versus 30-45 Hz power in the .dat EEG
ratios = []
for s in ["s01", "s12", "s24"]:
    raw = pickle.load(open(DAT / f"{s}.dat", "rb"), encoding="latin1")["data"][:, :32, BASE:BASE + 32 * FS]
    f_, p = signal.welch(raw, fs=FS, nperseg=256, axis=-1)
    hg = p[..., (f_ >= 45) & (f_ <= 60)].sum(-1); g = p[..., (f_ >= 30) & (f_ < 45)].sum(-1)
    ratios.append(np.median(hg / g))
say(f"6. median power ratio (45-60 Hz)/(30-45 Hz): {np.mean(ratios):.3f}")

Path("/kaggle/working/verify_data_report.txt").write_text("\n".join(out))
print("\nSaved /kaggle/working/verify_data_report.txt")

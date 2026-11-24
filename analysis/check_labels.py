"""
check_labels.py
===============
Verifies the emotion labels used in training against the official DEAP ratings.

Why: some redistributed copies of DEAP have valence/arousal ratings reflected
on part of the trials. If your copy is one of them, every result changes.

Checks, per participant:
  A. DEAP .dat labels  vs  official participant_ratings (matched by Experiment_id)
  B. EEG preprocessed labels (win_00.pt y)     vs  labels derived from the .dat
  C. Fusion preprocessed labels (win_00.pt y)  vs  labels derived from the .dat
  D. EEG and fusion datasets use the same train/val/test trials

Usage:
  python check_labels.py --dat_dir <folder with s01.dat ...> \
      --eeg_dir <pgcn-deap-preprocessed> --fus_dir <preprocessed_fusion> \
      --ratings <participant_ratings.csv or .xls>   (optional but strongly advised)
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

THR = 5.0                       # same threshold as preprocessing (rating > 5 = high)
COLS = ["Valence", "Arousal", "Dominance", "Liking"]


def binarize(r):
    r = np.asarray(r, float)
    v, a, d, l = (r[:, i] > THR for i in range(4))
    return np.stack([v, a, d, l, v * 2 + a], 1).astype(int)


def load_dat(dat_dir, subj):
    with open(Path(dat_dir) / f"{subj}.dat", "rb") as f:
        return pickle.load(f, encoding="latin1")["labels"].astype(float)   # (40, 4)


def load_official(path):
    p = Path(path)
    df = pd.read_excel(p) if p.suffix in (".xls", ".xlsx") else pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}
    need = ["participant_id", "experiment_id", "trial"] + [c.lower() for c in COLS]
    miss = [n for n in need if n not in low]
    if miss:
        raise SystemExit(f"Ratings file is missing columns {miss}. Found: {list(df.columns)}")
    return df.rename(columns={low[n]: n for n in need})


def preproc_labels(root, subj):
    """Binary labels [40, 5] stored in win_00.pt of each trial, plus meta."""
    sd = Path(root) / subj
    meta = json.load(open(sd / "meta.json"))
    ys = []
    for t in range(40):
        g = torch.load(sd / f"trial_{t:02d}" / "win_00.pt",
                       map_location="cpu", weights_only=False)
        ys.append(g.y.view(-1).numpy())
    return np.stack(ys).astype(int), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dat_dir", required=True)
    ap.add_argument("--eeg_dir", required=True)
    ap.add_argument("--fus_dir", required=True)
    ap.add_argument("--ratings", default=None)
    ap.add_argument("--out", default="label_check.csv")
    a = ap.parse_args()

    off = load_official(a.ratings) if a.ratings else None
    if off is None:
        print("!! No official ratings file given: check A is skipped.\n"
              "   Download metadata_csv.zip (participant_ratings.csv) from the DEAP\n"
              "   website, upload it as a Kaggle dataset, and pass --ratings.\n")

    rows = []
    for i in range(1, 33):
        s = f"s{i:02d}"
        if not (Path(a.dat_dir) / f"{s}.dat").exists():
            print(f"  {s}: .dat not found, skipped"); continue
        dat = load_dat(a.dat_dir, s)
        dat_bin = binarize(dat)
        r = {"subject": s}

        if off is not None:
            o = off[off.participant_id == i]
            o_by_exp = o.set_index("experiment_id").sort_index()
            o_by_trial = o.set_index("trial").sort_index()
            ref_exp = o_by_exp.loc[range(1, 41), [c.lower() for c in COLS]].values
            ref_trial = o_by_trial.loc[range(1, 41), [c.lower() for c in COLS]].values
            mism = np.any(np.abs(dat - ref_exp) > 0.01, axis=1)
            r["A_trials_rating_mismatch"] = int(mism.sum())
            r["A_valence_label_flips"] = int((binarize(ref_exp)[:, 0] != dat_bin[:, 0]).sum())
            r["A_arousal_label_flips"] = int((binarize(ref_exp)[:, 1] != dat_bin[:, 1]).sum())
            # diagnostic: does the .dat follow presentation order instead?
            r["A_mismatch_if_presentation_order"] = int(
                np.any(np.abs(dat - ref_trial) > 0.01, axis=1).sum())

        for tag, root in (("B_eeg", a.eeg_dir), ("C_fus", a.fus_dir)):
            if not (Path(root) / s / "meta.json").exists():
                r[f"{tag}_label_mismatch"] = None
                continue
            y, meta = preproc_labels(root, s)
            r[f"{tag}_label_mismatch"] = int(np.any(y != dat_bin, axis=1).sum())
            r[f"{tag}_meta"] = meta

        me, mf = r.pop("B_eeg_meta", None), r.pop("C_fus_meta", None)
        if me is not None and mf is not None:
            r["D_same_splits"] = all(sorted(me[k]) == sorted(mf[k])
                                     for k in ("train_trials", "val_trials", "test_trials"))
        rows.append(r)
        print("  " + "  ".join(f"{k}={v}" for k, v in r.items()), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    print("\n" + "=" * 70)
    problems = []
    for col in df.columns:
        if col.endswith("_mismatch") or col.endswith("_flips"):
            if col == "A_mismatch_if_presentation_order":
                continue
            bad = df[df[col].fillna(0) > 0]
            if len(bad):
                problems.append(f"{col}: {len(bad)} subjects, {int(bad[col].sum())} trials")
    if "D_same_splits" in df and (df.D_same_splits == False).any():
        problems.append(f"D_same_splits False for {list(df[df.D_same_splits == False].subject)}")
    if problems:
        print("  LABEL PROBLEMS FOUND. Do not use the results until this is fixed:")
        for p in problems:
            print("   -", p)
    else:
        print("  All checks passed" + ("" if off is not None
              else " (except A, which needs the official ratings file)"))
    print(f"  Table saved to {a.out}")


if __name__ == "__main__":
    main()

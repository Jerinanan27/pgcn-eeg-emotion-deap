"""
reevaluate_checkpoints.py
=========================
Re-score the saved best.pt checkpoints WITHOUT retraining.

Why: pgcn_experiment.evaluate() reports f1_score(average="binary"), which is the
F1 of the positive class only, not macro-F1. A model that predicts "high" for
every window scores 2p/(1+p) on that metric (about 0.69 to 0.92 for DEAP label
proportions), so collapse to the majority class looks like good performance.

For every subject or fold this script reports, per task:
  * f1_pos     : positive-class F1 (the metric used in the submitted paper)
  * macro_f1   : macro-averaged F1 over the classes
  * bal_acc    : balanced accuracy (chance = 0.5 binary, 0.25 quadrant)
  * acc        : plain accuracy
  * pred_pos   : fraction of windows/trials predicted as class 1 (collapse check)
at three levels:
  * window     : as in the paper
  * trial_mean : mean of the 29 window probabilities per trial, then argmax
  * trial_vote : majority vote over the 29 window predictions per trial
and a majority-class baseline (majority label of the training trials the model
saw: the subject's own 28 training trials within-subject, the training trials
of the other 31 subjects under LOSO), scored the same way.

It also saves the per-window probabilities to an .npz file so any later
statistic can be recomputed without touching the GPU again.

Usage (Kaggle, after copying the repo .py files to /kaggle/working):

  python reevaluate_checkpoints.py \
      --data_dir /kaggle/input/.../pgcn-deap-preprocessed \
      --ckpt_dir /kaggle/input/.../results_eeg/loso_none \
      --mode loso --fusion none --ablation none \
      --out_prefix /kaggle/working/reeval/loso_none

Run once per configuration (within/loso x none/early/late, plus each ablation).
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch_geometric.loader import DataLoader as GeoLoader

import pgcn_experiment as E
from pgcn_core import DEAPDataset

warnings.filterwarnings("ignore")

BIN = ["val", "aro", "dom", "lik"]
ALL = BIN + ["quad"]
NCLS = {"val": 2, "aro": 2, "dom": 2, "lik": 2, "quad": 4}


# ─── helpers ──────────────────────────────────────────────────────────────────

def list_units(ckpt_dir: Path, mode: str):
    """Subjects (within) or folds (loso) that have a saved checkpoint."""
    units = []
    for d in sorted(ckpt_dir.iterdir()):
        if not d.is_dir() or not (d / "best.pt").exists():
            continue
        if mode == "within" and d.name.startswith("s"):
            units.append((d.name, d / "best.pt"))
        elif mode == "loso" and (d.name.startswith("fold_") or d.name.startswith("s")):
            units.append((d.name.replace("fold_", ""), d / "best.pt"))
    return units


_LABEL_CACHE = {}


def trial_label_table(data_dir: Path, subj: str, trials):
    """Trial-level labels [n_trials, 5]; one window per trial is enough."""
    key = (str(data_dir), subj, tuple(trials))
    if key in _LABEL_CACHE:
        return _LABEL_CACHE[key]
    rows = []
    for t in trials:
        g = torch.load(data_dir / subj / f"trial_{t:02d}" / "win_00.pt",
                       map_location="cpu", weights_only=False)
        rows.append(g.y.view(-1).numpy())
    out = np.stack(rows) if rows else np.zeros((0, 5), dtype=int)
    _LABEL_CACHE[key] = out
    return out


def majority_labels(data_dir: Path, subjects):
    """Majority class per task over the TRAINING trials of `subjects`."""
    ys = []
    for s in subjects:
        meta = json.load(open(data_dir / s / "meta.json"))
        ys.append(trial_label_table(data_dir, s, meta["train_trials"]))
    Y = np.concatenate(ys)
    maj = {}
    for i, t in enumerate(ALL):
        counts = np.bincount(Y[:, i].astype(int), minlength=NCLS[t])
        maj[t] = int(np.argmax(counts))          # ties resolve to the lower class
    return maj


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    P = {t: [] for t in ALL}
    Y, T = [], []
    for b in loader:
        b = b.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(b)
        for t in ALL:
            P[t].append(torch.softmax(out[t].float(), dim=-1).cpu().numpy())
        Y.append(b.y.view(-1, 5).cpu().numpy())
        T.append(b.trial_idx.view(-1).cpu().numpy())
    return ({t: np.concatenate(v) for t, v in P.items()},
            np.concatenate(Y).astype(int), np.concatenate(T).astype(int))


def score(y_true, y_pred, task):
    labels = list(range(NCLS[task]))
    r = {
        "macro_f1": f1_score(y_true, y_pred, average="macro",
                             labels=labels, zero_division=0),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "acc": accuracy_score(y_true, y_pred),
        "n_true_classes": int(len(np.unique(y_true))),
        "n": int(len(y_true)),
    }
    if task in BIN:
        r["f1_pos"] = f1_score(y_true, y_pred, average="binary", zero_division=0)
        r["pred_pos"] = float(np.mean(y_pred == 1))
    return r


def to_levels(probs, Y, T):
    """Return {level: {task: (y_true, y_pred)}} for window and trial levels."""
    levels = {"window": {}, "trial_mean": {}, "trial_vote": {}}
    trials = np.unique(T)
    for i, t in enumerate(ALL):
        p = probs[t]
        levels["window"][t] = (Y[:, i], p.argmax(1))
        yt, ym, yv = [], [], []
        for tr in trials:
            m = T == tr
            yt.append(np.bincount(Y[m, i]).argmax())
            ym.append(p[m].mean(0).argmax())
            yv.append(np.bincount(p[m].argmax(1), minlength=NCLS[t]).argmax())
        levels["trial_mean"][t] = (np.array(yt), np.array(ym))
        levels["trial_vote"][t] = (np.array(yt), np.array(yv))
    return levels


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--mode", choices=["within", "loso"], required=True)
    ap.add_argument("--fusion", choices=["none", "early", "late"], default="none")
    ap.add_argument("--ablation", default="none",
                    choices=["none", "flat_gcn", "no_band_attn",
                             "struct_only", "no_hjorth"])
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()

    data_dir, ckpt_dir = Path(a.data_dir), Path(a.ckpt_dir)
    out_prefix = Path(a.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    no_hjorth = a.ablation == "no_hjorth"
    all_subjects = E.get_available_subjects(str(data_dir))

    units = list_units(ckpt_dir, a.mode)
    if not units:
        raise SystemExit(f"No best.pt checkpoints found under {ckpt_dir}")
    print(f"{len(units)} checkpoints  mode={a.mode} fusion={a.fusion} "
          f"ablation={a.ablation}  device={device}")

    rows, pooled, saved = [], {}, {}
    for subj, ckpt in units:
        split = "test" if a.mode == "within" else "all"
        ds = DEAPDataset(str(data_dir), [subj], split, no_hjorth)
        if len(ds) == 0:
            print(f"  [SKIP] {subj}: no test windows")
            continue
        dl = GeoLoader(ds, batch_size=a.bs, shuffle=False, num_workers=2)

        model = E.build_model(a.mode, a.fusion, a.ablation).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        probs, Y, T = predict(model, dl, device)
        saved[subj] = {"Y": Y, "T": T, **{f"p_{t}": probs[t] for t in ALL}}

        train_subjects = [subj] if a.mode == "within" else \
                         [s for s in all_subjects if s != subj]
        maj = majority_labels(data_dir, train_subjects)

        for level, by_task in to_levels(probs, Y, T).items():
            for t, (yt, yp) in by_task.items():
                yb = np.full_like(yt, maj[t])
                for system, pred in (("model", yp), ("majority", yb)):
                    r = score(yt, pred, t)
                    rows.append({"subject": subj, "level": level, "task": t,
                                 "system": system, **r})
                    key = (level, t, system)
                    pooled.setdefault(key, ([], []))
                    pooled[key][0].append(yt)
                    pooled[key][1].append(pred)

        m = [r for r in rows if r["subject"] == subj and r["level"] == "window"
             and r["system"] == "model" and r["task"] in BIN]

        # Sanity check: the window-level positive-class F1 must reproduce the
        # val_f1/aro_f1/... saved in metrics.json next to the checkpoint.
        # A mismatch means the wrong checkpoint, data folder or config.
        check = ""
        mpath = ckpt.parent / "metrics.json"
        if mpath.exists():
            old = json.load(open(mpath))
            diffs = [abs(r["f1_pos"] - old[f"{r['task']}_f1"])
                     for r in m if f"{r['task']}_f1" in old]
            if diffs:
                worst = max(diffs)
                check = f"  match_metrics_json={'OK' if worst < 0.01 else 'MISMATCH'}({worst:.3f})"
                for r in rows:
                    if r["subject"] == subj:
                        r["max_diff_vs_metrics_json"] = worst

        print(f"  {subj}  window pred_pos=" +
              " ".join(f"{r['task']}:{r['pred_pos']:.2f}" for r in m) +
              "  bal_acc=" + " ".join(f"{r['task']}:{r['bal_acc']:.2f}" for r in m) +
              check)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(f"{out_prefix}_per_subject.csv", index=False, float_format="%.4f")
    np.savez_compressed(f"{out_prefix}_window_probs.npz",
                        **{f"{s}__{k}": v for s, d in saved.items() for k, v in d.items()})

    # Pooled metrics: concatenate all subjects' predictions, then score once.
    # Use these for within-subject, where 6 test trials per subject make
    # per-subject macro-F1 unstable or undefined.
    prow = []
    for (level, t, system), (yts, yps) in pooled.items():
        r = score(np.concatenate(yts), np.concatenate(yps), t)
        prow.append({"level": level, "task": t, "system": system, **r})
    pdf = pd.DataFrame(prow)
    pdf.to_csv(f"{out_prefix}_pooled.csv", index=False, float_format="%.4f")

    cols = ["f1_pos", "macro_f1", "bal_acc", "pred_pos"]
    summ = (df.groupby(["level", "task", "system"])[cols]
              .agg(["mean", "std"]).round(3))
    summ.to_csv(f"{out_prefix}_summary.csv")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print("\nMean across subjects (window level):")
        print(summ.loc["window"])
        print("\nMean across subjects (trial_mean level):")
        print(summ.loc["trial_mean"])
    print(f"\nSaved {out_prefix}_per_subject.csv, _pooled.csv, _summary.csv, "
          f"_window_probs.npz")


if __name__ == "__main__":
    main()

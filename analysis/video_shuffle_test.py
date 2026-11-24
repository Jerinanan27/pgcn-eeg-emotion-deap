"""
video_shuffle_test.py
=====================
Test-time video manipulation on the trained WITHIN-SUBJECT fusion models
(R2.2, inference only, no training).

For each subject's 6 test trials the EEG input is unchanged and only the video
embedding is replaced:
  original      : the trial's own video embedding
  other_trial   : the embedding of a randomly chosen DIFFERENT trial of the same
                  subject (repeated n_rep times, results averaged)
  subject_mean  : the subject's mean embedding over its training trials
                  (same identity, no trial-specific information)

If the within-subject fusion gain depends on trial-specific video content,
other_trial and subject_mean should fall back towards EEG-only performance.

Batch size 32 matches training (BandAttn uses batch statistics).

Usage:
  python video_shuffle_test.py --fus_dir <preprocessed_fusion> \
      --ckpt_root <.../results_fusion> --out_dir shuffle [--n_rep 20]
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch_geometric.loader import DataLoader as GeoLoader

import pgcn_experiment as E
from pgcn_core import DEAPDataset

warnings.filterwarnings("ignore")
TASKS = ["val", "aro", "dom", "lik", "quad"]
NCLS = {"val": 2, "aro": 2, "dom": 2, "lik": 2, "quad": 4}


def video_table(fus_dir, subj):
    """[40, 768] video embedding per trial (constant across windows)."""
    vids = []
    for t in range(40):
        g = torch.load(Path(fus_dir) / subj / f"trial_{t:02d}" / "win_00.pt",
                       map_location="cpu", weights_only=False)
        vids.append(g.video.view(-1).float())
    return torch.stack(vids)


@torch.no_grad()
def predict(model, loader, device, vid_for_trial):
    """vid_for_trial: tensor [40, 768] giving the video to feed for each trial."""
    model.eval()
    P = {t: [] for t in TASKS}
    Y, T = [], []
    vt = vid_for_trial.to(device)
    for b in loader:
        b = b.to(device, non_blocking=True)
        tidx = b.trial_idx.view(-1).long()
        b.video = vt[tidx]
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(b)
        for t in TASKS:
            P[t].append(torch.softmax(out[t].float(), -1).cpu().numpy())
        Y.append(b.y.view(-1, 5).cpu().numpy())
        T.append(tidx.cpu().numpy())
    return ({t: np.concatenate(v) for t, v in P.items()},
            np.concatenate(Y).astype(int), np.concatenate(T))


def scores(P, Y, T):
    r = {}
    trials = np.unique(T)
    for i, t in enumerate(TASKS):
        yw, pw = Y[:, i], P[t].argmax(1)
        r[f"{t}_bacc_window"] = balanced_accuracy_score(yw, pw)
        r[f"{t}_macroF1_window"] = f1_score(yw, pw, average="macro",
                                            labels=list(range(NCLS[t])), zero_division=0)
        yt = np.array([np.bincount(Y[T == k, i]).argmax() for k in trials])
        pt = np.array([P[t][T == k].mean(0).argmax() for k in trials])
        r[f"{t}_bacc_trial"] = balanced_accuracy_score(yt, pt)
    return r


def bh(p):
    p = np.asarray(p, float); m = len(p)
    o = np.argsort(p); r = p[o] * m / np.arange(1, m + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(r, 1); return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fus_dir", required=True)
    ap.add_argument("--ckpt_root", required=True,
                    help="folder containing within_early/ and within_late/")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_rep", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)

    rows = []
    for fusion in ("early", "late"):
        cdir = Path(a.ckpt_root) / f"within_{fusion}"
        subs = sorted(d.name for d in cdir.iterdir()
                      if d.is_dir() and (d / "best.pt").exists())
        print(f"\nwithin_{fusion}: {len(subs)} checkpoints")
        for s in subs:
            meta = json.load(open(Path(a.fus_dir) / s / "meta.json"))
            ds = DEAPDataset(a.fus_dir, [s], "test")
            dl = GeoLoader(ds, batch_size=a.bs, shuffle=False, num_workers=2)
            model = E.build_model("within", fusion, "none").to(device)
            model.load_state_dict(torch.load(cdir / s / "best.pt",
                                             map_location=device, weights_only=True))
            V = video_table(a.fus_dir, s)
            rng = np.random.default_rng(int(s[1:]))

            # original
            r0 = scores(*predict(model, dl, device, V))
            rows.append({"fusion": fusion, "subject": s, "condition": "original", **r0})

            # subject mean over training trials
            vm = V[meta["train_trials"]].mean(0, keepdim=True).expand(40, -1)
            rm = scores(*predict(model, dl, device, vm))
            rows.append({"fusion": fusion, "subject": s, "condition": "subject_mean", **rm})

            # other trial of the same subject (never its own)
            reps = []
            for _ in range(a.n_rep):
                src = np.array([rng.choice([k for k in range(40) if k != t])
                                for t in range(40)])
                reps.append(scores(*predict(model, dl, device, V[src])))
            ro = pd.DataFrame(reps).mean().to_dict()
            rows.append({"fusion": fusion, "subject": s, "condition": "other_trial", **ro})

            print(f"  {s}  valence bacc: original={r0['val_bacc_window']:.2f} "
                  f"other_trial={ro['val_bacc_window']:.2f} "
                  f"subject_mean={rm['val_bacc_window']:.2f}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(out / "video_shuffle_per_subject.csv", index=False, float_format="%.4f")

    # paired tests: original vs each manipulation
    tests = []
    for fusion in ("early", "late"):
        d = df[df.fusion == fusion]
        o = d[d.condition == "original"].set_index("subject")
        for cond in ("other_trial", "subject_mean"):
            c = d[d.condition == cond].set_index("subject").loc[o.index]
            for t in TASKS:
                for lvl in ("window", "trial"):
                    col = f"{t}_bacc_{lvl}"
                    diff = o[col].values - c[col].values
                    p = 1.0 if np.allclose(diff, 0) else wilcoxon(o[col], c[col]).pvalue
                    tests.append({"fusion": fusion, "manipulation": cond, "task": t,
                                  "level": lvl, "original": o[col].mean(),
                                  "manipulated": c[col].mean(),
                                  "drop": diff.mean(), "p": p})
    tt = pd.DataFrame(tests)
    tt["p_bh"] = bh(tt.p.values)
    tt.to_csv(out / "video_shuffle_tests.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print("\nOriginal vs manipulated video (window-level balanced accuracy):")
        print(tt[tt.level == "window"].round(3).to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

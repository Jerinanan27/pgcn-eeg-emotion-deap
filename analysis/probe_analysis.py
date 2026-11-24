"""
probe_analysis.py
=================
Linear probes on the frozen ViT-B/16 trial embeddings (R2.2, no network training).

For the 22 subjects with real video, one 768-d embedding per trial.
Two embedding versions:
  raw      : as used by the fusion models
  centred  : minus that subject's mean embedding over its own TRAINING trials
             (label-free; the same information EEG normalisation already uses)

Probes (logistic regression, class-balanced, standardised inputs):
  P1 identity  : which of the 22 subjects? (5-fold CV over trials; chance 1/22)
  P2 seen-subject / unseen-trial : train on train+val trials of all 22 subjects,
     test on their test trials (the within-subject split)
  P3 unseen-subject (LOSO) : train on 21 subjects, test on all 40 trials of the
     held-out subject
For P2 and P3: balanced accuracy per subject and pooled, a majority baseline,
and a label-permutation null (labels shuffled within each training subject).

Optional drift check (needs official participant_ratings.csv): do embeddings
encode presentation position in the session, and do labels drift with it?
If both hold, part of any within-subject video gain may reflect session drift.

Usage:
  python probe_analysis.py --fus_dir <preprocessed_fusion> --out_dir probes \
      [--ratings participant_ratings.csv] [--n_perm 100] [--n_jobs -1]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TASKS = {"val": 0, "aro": 1, "dom": 2, "lik": 3, "quad": 4}


# ─── data ─────────────────────────────────────────────────────────────────────

def load_embeddings(fus_dir):
    import torch
    root = Path(fus_dir)
    X, Y, S, T, split = [], [], [], [], []
    for i in range(1, 33):
        s = f"s{i:02d}"
        mp = root / s / "meta.json"
        if not mp.exists():
            continue
        meta = json.load(open(mp))
        if not meta.get("has_real_video", False):
            continue
        sp = {t: "train" for t in meta["train_trials"]}
        sp.update({t: "val" for t in meta["val_trials"]})
        sp.update({t: "test" for t in meta["test_trials"]})
        for t in range(40):
            g = torch.load(root / s / f"trial_{t:02d}" / "win_00.pt",
                           map_location="cpu", weights_only=False)
            X.append(g.video.view(-1).float().numpy())
            Y.append(g.y.view(-1).numpy())
            S.append(s); T.append(t); split.append(sp[t])
    return (np.stack(X), np.stack(Y).astype(int), np.array(S),
            np.array(T), np.array(split))


def centre(X, S, split):
    """Subtract each subject's mean over its own training trials."""
    Xc = X.copy()
    for s in np.unique(S):
        m = S == s
        Xc[m] -= X[m & (split == "train")].mean(0)
    return Xc


def clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=3000,
                                            class_weight="balanced"))


# ─── probes ───────────────────────────────────────────────────────────────────

def identity_probe(X, S, seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = cross_val_predict(clf(), X, S, cv=cv)
    return float(np.mean(pred == S))


def per_subject_bacc(y, p, S):
    vals = []
    for s in np.unique(S):
        m = S == s
        if len(np.unique(y[m])) < 2:        # balanced accuracy undefined-ish
            vals.append(np.nan); continue
        vals.append(balanced_accuracy_score(y[m], p[m]))
    return float(np.nanmean(vals)), int(np.sum(~np.isnan(vals)))


def shuffle_within(y, S, rng):
    y = y.copy()
    for s in np.unique(S):
        m = np.where(S == s)[0]
        y[m] = y[rng.permutation(m)]
    return y


def run_seen(X, y, S, split, y_train=None):
    tr, te = split != "test", split == "test"
    yt = y if y_train is None else y_train
    p = clf().fit(X[tr], yt[tr]).predict(X[te])
    return p, y[te], S[te]


def run_loso(X, y, S, y_train=None):
    yt = y if y_train is None else y_train
    P = np.empty_like(y)
    for s in np.unique(S):
        te = S == s
        P[te] = clf().fit(X[~te], yt[~te]).predict(X[te])
    return P, y, S


def majority_pred(y, S, split, mode):
    if mode == "seen":
        tr, te = split != "test", split == "test"
        p = np.full(te.sum(), np.bincount(y[tr]).argmax())
        return p, y[te], S[te]
    P = np.empty_like(y)
    for s in np.unique(S):
        te = S == s
        P[te] = np.bincount(y[~te]).argmax()
    return P, y, S


def evaluate(p, y, S):
    ps, n = per_subject_bacc(y, p, S)
    return {"bacc_per_subject": ps, "n_subjects_scored": n,
            "bacc_pooled": float(balanced_accuracy_score(y, p))}


def perm_null(X, y, S, split, mode, n_perm, n_jobs, seed=0):
    def one(k):
        rng = np.random.default_rng(seed + k)
        ys = shuffle_within(y, S, rng)
        if mode == "seen":
            p, yy, ss = run_seen(X, y, S, split, y_train=ys)
        else:
            p, yy, ss = run_loso(X, y, S, y_train=ys)
        return balanced_accuracy_score(yy, p)
    return np.array(Parallel(n_jobs=n_jobs)(delayed(one)(k) for k in range(n_perm)))


# ─── drift check ──────────────────────────────────────────────────────────────

def drift_check(X, Y, S, T, ratings_path):
    p = Path(ratings_path)
    r = pd.read_excel(p) if p.suffix in (".xls", ".xlsx") else pd.read_csv(p)
    r.columns = [c.strip().lower() for c in r.columns]
    rows = []
    for s in np.unique(S):
        pid = int(s[1:])
        o = r[r.participant_id == pid].set_index("experiment_id")["trial"]
        m = S == s
        pos = o.loc[T[m] + 1].values.astype(float)        # trial index t = video t+1
        Xs = X[m]

        def cv_rho(target, seed):
            model = make_pipeline(StandardScaler(),
                                  RidgeCV(alphas=np.logspace(-1, 4, 12)))
            pred = cross_val_predict(model, Xs, target,
                                     cv=KFold(5, shuffle=True, random_state=seed))
            return spearmanr(pred, target).correlation

        rho_emb = cv_rho(pos, 0)
        # cross-validated regression is biased below 0 under the null, so the
        # reference is the same statistic with presentation positions shuffled
        rng = np.random.default_rng(int(s[1:]))
        null = [cv_rho(rng.permutation(pos), k) for k in range(20)]
        row = {"subject": s, "rho_embedding_predicts_position": rho_emb,
               "rho_null_mean": float(np.mean(null)),
               "rho_null_95": float(np.percentile(null, 95))}
        for t, k in TASKS.items():
            if t == "quad":
                continue
            yv = Y[m, k]
            row[f"rho_{t}_label_vs_position"] = (spearmanr(yv, pos).correlation
                                                 if len(np.unique(yv)) > 1 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fus_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ratings", default=None)
    ap.add_argument("--n_perm", type=int, default=100)
    ap.add_argument("--n_jobs", type=int, default=-1)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    X, Y, S, T, split = load_embeddings(a.fus_dir)
    print(f"Embeddings: {X.shape}  subjects={len(np.unique(S))}")
    versions = {"raw": X, "centred": centre(X, S, split)}

    # P1 identity
    rows = [{"version": v, "identity_acc": identity_probe(Xv, S),
             "chance": 1 / len(np.unique(S))} for v, Xv in versions.items()]
    pd.DataFrame(rows).to_csv(out / "P1_identity.csv", index=False, float_format="%.4f")
    print("\nP1 identity probe"); print(pd.DataFrame(rows).round(3).to_string(index=False))

    # P2 / P3 affect probes
    res = []
    for mode in ("seen", "loso"):
        for t, k in TASKS.items():
            y = Y[:, k]
            mp, my, ms = majority_pred(y, S, split, mode)
            base = evaluate(mp, my, ms)
            for v, Xv in versions.items():
                p, yy, ss = (run_seen(Xv, y, S, split) if mode == "seen"
                             else run_loso(Xv, y, S))
                r = {"probe": "P2_seen_subject" if mode == "seen" else "P3_unseen_subject",
                     "task": t, "version": v, **evaluate(p, yy, ss),
                     "majority_bacc_pooled": base["bacc_pooled"]}
                if t in ("val", "aro", "quad") and a.n_perm > 0:
                    null = perm_null(Xv, y, S, split, mode, a.n_perm, a.n_jobs)
                    r["perm_null_mean"] = float(null.mean())
                    r["perm_null_95"] = float(np.percentile(null, 95))
                    r["perm_p"] = float((1 + np.sum(null >= r["bacc_pooled"])) / (1 + len(null)))
                res.append(r)
                print(f"  {r['probe']:18s} {t:5s} {v:8s} pooled bacc={r['bacc_pooled']:.3f} "
                      f"per-subj={r['bacc_per_subject']:.3f} "
                      f"majority={r['majority_bacc_pooled']:.3f}"
                      + (f"  perm_p={r['perm_p']:.3f}" if "perm_p" in r else ""), flush=True)
    pd.DataFrame(res).to_csv(out / "P2_P3_affect_probes.csv", index=False, float_format="%.4f")

    # drift check
    if a.ratings:
        d = drift_check(versions["raw"], Y, S, T, a.ratings)
        d.to_csv(out / "drift_check.csv", index=False, float_format="%.4f")
        print("\nDrift check (Spearman, mean over subjects):")
        print(d.drop(columns="subject").mean().round(3).to_string())
        n_above = int((d.rho_embedding_predicts_position > d.rho_null_95).sum())
        print(f"  subjects whose embeddings predict session position above their "
              f"95% null: {n_above}/{len(d)}")
        print("  Compare rho_embedding_predicts_position with rho_null_mean, not with 0.\n"
              "  If embeddings track position AND labels also track position, session\n"
              "  drift could explain part of the within-subject video gain.")
    else:
        print("\n(Drift check skipped: pass --ratings participant_ratings.csv)")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

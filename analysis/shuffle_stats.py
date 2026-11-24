"""
shuffle_stats.py
================
Paired tests for the test-time video manipulation (output of
video_shuffle_test.py), recomputed from the per-participant scores so that the
same absent-class rule as stats_analysis.py can be applied.

A participant-task pair is excluded when that participant's six within-subject
test trials lack one of the task's classes (taken from the n_true_classes
column written by reevaluate_checkpoints.py for the same test trials).

Tests: two-sided Wilcoxon signed-rank (original vs manipulated) over
participants, Benjamini-Hochberg correction over all 40 tests
(2 fusion types x 2 manipulations x 5 tasks x 2 levels).

Usage:
  python shuffle_stats.py --shuffle_csv shuffle/video_shuffle_per_subject.csv \
      --reeval_dir reeval_bs --out shuffle/video_shuffle_tests_complete.csv [--all]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

TASKS = ["val", "aro", "dom", "lik", "quad"]
NCLS = {"val": 2, "aro": 2, "dom": 2, "lik": 2, "quad": 4}


def bh(p):
    p = np.asarray(p, float); m = len(p)
    o = np.argsort(p); r = p[o] * m / np.arange(1, m + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(r, 1); return out


def complete_subjects(reeval_dir, fusion, task):
    d = pd.read_csv(Path(reeval_dir) / f"within_{fusion}_per_subject.csv")
    d = d[(d.level == "window") & (d.system == "model") & (d.task == task)]
    return set(d[d.n_true_classes >= NCLS[task]].subject)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffle_csv", required=True)
    ap.add_argument("--reeval_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--all", action="store_true", help="keep all participants")
    a = ap.parse_args()
    df = pd.read_csv(a.shuffle_csv)
    rows = []
    for fusion in ("early", "late"):
        d = df[df.fusion == fusion]
        o = d[d.condition == "original"].set_index("subject")
        for cond in ("other_trial", "subject_mean"):
            c = d[d.condition == cond].set_index("subject")
            for t in TASKS:
                subs = sorted(o.index if a.all else
                              set(o.index) & complete_subjects(a.reeval_dir, fusion, t))
                for lvl in ("window", "trial"):
                    col = f"{t}_bacc_{lvl}"
                    x, y = o.loc[subs, col].values, c.loc[subs, col].values
                    p = 1.0 if np.allclose(x, y) else wilcoxon(x, y).pvalue
                    rows.append({"fusion": fusion, "manipulation": cond, "task": t,
                                 "level": lvl, "n": len(subs), "original": x.mean(),
                                 "manipulated": y.mean(), "drop": (x - y).mean(), "p": p})
    tt = pd.DataFrame(rows)
    tt["p_bh"] = bh(tt.p.values)
    tt.to_csv(a.out, index=False, float_format="%.6f")
    print(tt[tt.level == "window"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""
stats_analysis.py
=================
Statistics on the re-scored checkpoints (output of reevaluate_checkpoints.py).
CPU only, no PyTorch needed.

Three families of paired, two-sided Wilcoxon signed-rank tests over subjects,
each corrected with Benjamini-Hochberg FDR inside its own family:

  F1  model vs majority-class baseline       (every config x task)
  F2  fusion vs EEG-only, matched subjects   (s01-s22, within and LOSO)
  F3  ablation vs full PGCN                  (within and LOSO)

Primary metric: balanced accuracy at window level. The same tests are repeated
for macro-F1 and at trial level (mean of the 29 window probabilities).

Absent classes: a participant's test set can lack a class (for example all six
within-subject test trials rated high in liking, or a quadrant with no test
trial). Balanced accuracy then averages recall over the classes that are
present only, so a constant predictor no longer scores 0.50 (or 0.25), and the
majority baseline can score 0.0 or 1.0. With --complete_classes, every
participant-task pair whose test set lacks a class is excluded from that task's
analyses (the excluded pairs are listed in excluded_pairs.csv). Without the
flag all participants are kept (sensitivity analysis).

Usage:
  python stats_analysis.py --in_dir reeval_bs --out_dir stats_complete --complete_classes
  python stats_analysis.py --in_dir reeval_bs --out_dir stats_all
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

TASKS = ["val", "aro", "dom", "lik", "quad"]
NCLS = {"val": 2, "aro": 2, "dom": 2, "lik": 2, "quad": 4}
TASK_NAME = {"val": "Valence", "aro": "Arousal", "dom": "Dominance",
             "lik": "Liking", "quad": "Quadrant"}
ABL_ALIAS = {"no_band": "no_band_attn"}          # folder names differ by protocol


def bh(p):
    """Benjamini-Hochberg adjusted p-values (NaNs kept as NaN)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = ok.sum()
    if m == 0:
        return out
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(ranked, 1.0)
    out[ok] = adj
    return out


def boot_ci(d, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    if len(d) < 2:
        return np.nan, np.nan
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_test(a, b):
    """Wilcoxon a vs b. Returns dict with means, diff, CI, p, effect size."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    nz = int(np.sum(np.abs(d) > 1e-12))
    if nz == 0:
        p, rbc = 1.0, 0.0
    else:
        try:
            p = float(wilcoxon(a, b, zero_method="wilcox",
                               alternative="two-sided").pvalue)
        except ValueError:
            p = 1.0
        dn = d[np.abs(d) > 1e-12]
        ranks = pd.Series(np.abs(dn)).rank().values
        rbc = float((ranks[dn > 0].sum() - ranks[dn < 0].sum()) / ranks.sum())
    lo, hi = boot_ci(d)
    return {"n": len(d), "n_nonzero": nz, "mean_a": a.mean(), "mean_b": b.mean(),
            "mean_diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
            "p": p, "rank_biserial": rbc}


def load(in_dir: Path):
    frames = []
    for f in sorted(in_dir.glob("*_per_subject.csv")):
        name = f.name.replace("_per_subject.csv", "")
        mode, rest = name.split("_", 1)
        df = pd.read_csv(f)
        df["config"] = name
        df["mode"] = mode
        if rest in ("early", "late"):
            df["fusion"], df["ablation"] = rest, "none"
        else:
            df["fusion"], df["ablation"] = "none", ABL_ALIAS.get(rest, rest)
        frames.append(df)
    if not frames:
        raise SystemExit(f"No *_per_subject.csv in {in_dir}")
    return pd.concat(frames, ignore_index=True)


def drop_incomplete(df):
    """Remove participant-task-level rows whose test set lacks a class."""
    k = df.task.map(NCLS)
    bad = df.n_true_classes < k
    excl = (df[bad][["config", "subject", "level", "task", "n_true_classes"]]
            .drop_duplicates())
    return df[~bad].copy(), excl


def pick(df, config, level, task, system, metric):
    s = df[(df.config == config) & (df.level == level) &
           (df.task == task) & (df.system == system)]
    return s.set_index("subject")[metric]


def family_vs_majority(df, level, metric):
    rows = []
    for cfg in sorted(df.config.unique()):
        for t in TASKS:
            m = pick(df, cfg, level, t, "model", metric)
            b = pick(df, cfg, level, t, "majority", metric)
            subs = m.index.intersection(b.index)
            if len(subs) == 0:
                continue
            r = paired_test(m[subs], b[subs])
            rows.append({"config": cfg, "task": t, **r})
    out = pd.DataFrame(rows)
    out["p_bh"] = bh(out.p)
    return out


def family_pairs(df, pairs, level, metric, matched_to=None):
    rows = []
    for a_cfg, b_cfg in pairs:
        if a_cfg not in df.config.values or b_cfg not in df.config.values:
            continue
        for t in TASKS:
            a = pick(df, a_cfg, level, t, "model", metric)
            b = pick(df, b_cfg, level, t, "model", metric)
            subs = a.index.intersection(b.index)
            if matched_to is not None:
                subs = subs.intersection(matched_to)
            r = paired_test(a[sorted(subs)], b[sorted(subs)])
            rows.append({"a": a_cfg, "b": b_cfg, "task": t, **r})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_bh"] = bh(out.p)
    return out


def descriptive(df):
    cols = ["bal_acc", "macro_f1", "f1_pos", "pred_pos"]
    g = (df.groupby(["config", "level", "task", "system"])[cols]
           .agg(["mean", "std"]))
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.reset_index()


def fmt(x, nd=3):
    return "" if pd.isna(x) else f"{x:.{nd}f}"


def stars(p):
    if pd.isna(p):
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--complete_classes", action="store_true",
                    help="exclude participant-task pairs whose test set lacks a class")
    a = ap.parse_args()
    in_dir, out_dir = Path(a.in_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(in_dir)
    print(f"Loaded {df.config.nunique()} configs: {sorted(df.config.unique())}")
    if a.complete_classes:
        df, excl = drop_incomplete(df)
        excl.to_csv(out_dir / "excluded_pairs.csv", index=False)
        summ = (excl[excl.level == "window"].drop_duplicates(["config", "subject", "task"])
                .assign(mode=lambda x: x.config.str.split("_").str[0])
                .drop_duplicates(["mode", "subject", "task"])
                .groupby(["mode", "task"]).subject.apply(lambda x: ",".join(sorted(x))))
        print("\nExcluded participant-task pairs (test set lacks a class):")
        print(summ.to_string())

    if "max_diff_vs_metrics_json" in df.columns:
        md = df.groupby("config")["max_diff_vs_metrics_json"].max()
        print("\nLargest difference vs saved metrics.json per config "
              "(should be < 0.01 once batch sizes match training):")
        print(md.round(3).to_string())

    descriptive(df).to_csv(out_dir / "descriptive.csv", index=False,
                           float_format="%.6f")

    fusion_subs = pd.Index(sorted(df[df.fusion != "none"].subject.unique()))
    fusion_pairs = [("within_early", "within_none"), ("within_late", "within_none"),
                    ("within_late", "within_early"),
                    ("loso_early", "loso_none"), ("loso_late", "loso_none")]
    abl_pairs = []
    for mode in ("within", "loso"):
        for c in sorted(df[(df["mode"] == mode) & (df.fusion == "none")
                           & (df.ablation != "none")].config.unique()):
            abl_pairs.append((c, f"{mode}_none"))

    report = []
    for level in ("window", "trial_mean"):
        for metric in ("bal_acc", "macro_f1"):
            tag = f"{level}_{metric}"
            f1 = family_vs_majority(df, level, metric)
            f2 = family_pairs(df, fusion_pairs, level, metric, matched_to=fusion_subs)
            f3 = family_pairs(df, abl_pairs, level, metric)
            f1.to_csv(out_dir / f"F1_vs_majority_{tag}.csv", index=False, float_format="%.6f")
            f2.to_csv(out_dir / f"F2_fusion_vs_eeg_{tag}.csv", index=False, float_format="%.6f")
            f3.to_csv(out_dir / f"F3_ablation_{tag}.csv", index=False, float_format="%.6f")
            report.append((tag, f1, f2, f3))

    # pooled within-subject metrics (6 test trials per subject is too few alone)
    pooled = []
    for f in sorted(in_dir.glob("within_*_pooled.csv")):
        p = pd.read_csv(f)
        p["config"] = f.name.replace("_pooled.csv", "")
        pooled.append(p)
    if pooled:
        pooled = pd.concat(pooled)
        pooled.to_csv(out_dir / "pooled_within.csv", index=False, float_format="%.6f")

    # ---------- readable report (primary analysis) ----------
    lines = []
    for tag, f1, f2, f3 in report:
        lines.append(f"\n{'=' * 78}\n  {tag.upper()}\n{'=' * 78}")
        lines.append("\nF1  model vs majority baseline (Wilcoxon, BH over all configs x tasks)")
        lines.append(f"  {'config':22s}{'task':6s}{'n':>4s}{'model':>8s}{'major':>8s}"
                     f"{'diff':>8s}{'95% CI':>17s}{'p_BH':>9s}")
        for _, r in f1.iterrows():
            lines.append(f"  {r.config:22s}{r.task:6s}{r.n:4d}{fmt(r.mean_a):>8s}"
                         f"{fmt(r.mean_b):>8s}{fmt(r.mean_diff):>8s}"
                         f"  [{fmt(r.ci_lo)},{fmt(r.ci_hi)}]{fmt(r.p_bh, 4):>9s} {stars(r.p_bh)}")
        lines.append("\nF2  fusion vs EEG-only, matched subjects (BH over this family)")
        for _, r in f2.iterrows():
            lines.append(f"  {r.a:14s} vs {r.b:14s}{r.task:6s}{r.n:4d}{fmt(r.mean_a):>8s}"
                         f"{fmt(r.mean_b):>8s}{fmt(r.mean_diff):>8s}"
                         f"  [{fmt(r.ci_lo)},{fmt(r.ci_hi)}]{fmt(r.p_bh, 4):>9s} {stars(r.p_bh)}")
        lines.append("\nF3  ablation vs full PGCN (BH over this family)")
        for _, r in f3.iterrows():
            lines.append(f"  {r.a:22s} vs {r.b:12s}{r.task:6s}{r.n:4d}{fmt(r.mean_a):>8s}"
                         f"{fmt(r.mean_b):>8s}{fmt(r.mean_diff):>8s}"
                         f"  [{fmt(r.ci_lo)},{fmt(r.ci_hi)}]{fmt(r.p_bh, 4):>9s} {stars(r.p_bh)}")
    text = "\n".join(lines)
    (out_dir / "report.txt").write_text(text)
    print(text)
    print(f"\nSaved all tables to {out_dir}")


if __name__ == "__main__":
    main()

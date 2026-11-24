"""
make_figures.py
===============
Generates the four result figures of the manuscript from the CSV files in
revision_results.zip (output of the Kaggle revision notebook). CPU only.

  figures/fig3_within.pdf    within-participant balanced accuracy, matched s01-s22
  figures/fig4_loso.pdf      LOSO balanced accuracy + positive-class F1 vs a
                             constant "always high" predictor
  figures/fig5_shuffle.pdf   test-time video manipulation (within-participant fusion)
  figures/fig6_probes.pdf    linear probes on the frozen ViT-B/16 embeddings

Usage:
  pip install pandas numpy matplotlib
  python make_figures.py --res revision_results --out figures

Requires: pandas, numpy, matplotlib (no PyTorch).

As in make_tables.py, a participant-task pair is left out when the participant's
test set lacks one of the task's classes (balanced accuracy would otherwise be
averaged over the classes present only). Statistics come from stats_complete/.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TASKS = ["val", "aro", "dom", "lik", "quad"]
TNAME = {"val": "Valence", "aro": "Arousal", "dom": "Dominance",
         "lik": "Liking", "quad": "Quadrant"}
MODELS = [("none", "EEG-only"), ("early", "Early fusion"), ("late", "Late fusion")]
# Okabe-Ito colours (colour-blind safe; validated as a 3-slot categorical set)
COL = {"none": "#0072B2", "early": "#D55E00", "late": "#009E73"}
GREY_D, GREY_L, INK, MUTED = "#4D4D4D", "#9A9A9A", "#1A1A1A", "#6B6B6B"
FUSION_SUBJECTS = [f"s{i:02d}" for i in range(1, 23)]
NCLS = {"val": 2, "aro": 2, "dom": 2, "lik": 2, "quad": 4}
MM = 1 / 25.4  # inches per mm (Scientific Reports: 89 mm single, 183 mm double column)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
})


# ─── helpers ──────────────────────────────────────────────────────────────────

def per_subject(res, config, level="window"):
    df = pd.read_csv(res / "reeval_bs" / f"{config}_per_subject.csv")
    df = df[df.level == level]
    return df[df.n_true_classes >= df.task.map(NCLS)]      # complete test sets only


def values(df, task, system="model", metric="bal_acc", subjects=None):
    s = df[(df.task == task) & (df.system == system)].set_index("subject")[metric]
    if subjects is not None:
        s = s.reindex([x for x in subjects if x in s.index])
    return s


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def stars(p):
    if pd.isna(p) or p >= 0.05:
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*"


def style_axis(ax, ylim, chance=None):
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.5)
    ax.set_axisbelow(True)
    if chance is not None:
        ax.axhline(chance, color=GREY_L, linewidth=0.6, linestyle=":", zorder=0)


def model_panel(ax, data, task, baseline, ylim, chance, pmap=None):
    """Per-subject dots + mean with 95% bootstrap CI for each model."""
    rng = np.random.default_rng(1)
    for i, (key, label) in enumerate(MODELS):
        if key not in data:
            continue
        v = data[key].values
        jit = rng.uniform(-0.14, 0.14, len(v))
        ax.scatter(np.full(len(v), i) + jit, v, s=5, color=COL[key], alpha=0.35,
                   linewidths=0, zorder=2)
        lo, hi = boot_ci(v)
        ax.errorbar(i + 0.30, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]],
                    fmt="o", ms=3.6, color=COL[key], mec="white", mew=0.6,
                    elinewidth=1.0, capsize=0, zorder=3)
        if pmap and key in pmap and stars(pmap[key]):
            ax.text(i + 0.30, hi + 0.03 * (ylim[1] - ylim[0]), stars(pmap[key]),
                    ha="center", va="bottom", fontsize=7, color=INK)
    ax.axhline(baseline, color=GREY_D, linewidth=0.8, linestyle="--", zorder=1)
    style_axis(ax, ylim, chance)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(["EEG", "Early", "Late"])
    ax.set_xlim(-0.5, len(MODELS) - 0.3)
    ax.set_title(TNAME[task], color=INK)


def legend_handles():
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=COL[k], ms=4, label=l) for k, l in MODELS]
    h.append(Line2D([], [], color=GREY_D, ls="--", lw=0.8, label="Majority-class baseline"))
    return h


# ─── Figure 3: within-participant ─────────────────────────────────────────────────

def fig_within(res, out):
    dfs = {k: per_subject(res, f"within_{k}") for k, _ in MODELS}
    f2 = pd.read_csv(res / "stats_complete" / "F2_fusion_vs_eeg_window_bal_acc.csv")
    fig, axes = plt.subplots(1, 5, figsize=(183 * MM, 58 * MM), sharey=False)
    for ax, t in zip(axes, TASKS):
        data = {k: values(dfs[k], t, subjects=FUSION_SUBJECTS) for k, _ in MODELS}
        base = values(dfs["early"], t, system="majority", subjects=FUSION_SUBJECTS).mean()
        pmap = {}
        for k in ("early", "late"):
            r = f2[(f2.a == f"within_{k}") & (f2.b == "within_none") & (f2.task == t)]
            if len(r):
                pmap[k] = r.p_bh.iloc[0]
        ylim = (0.0, 1.05)
        model_panel(ax, data, t, base, ylim, 0.25 if t == "quad" else 0.5, pmap)
    axes[0].set_ylabel("Balanced accuracy (window level)")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.6)
    fig.savefig(out / "fig3_within.pdf", bbox_inches="tight")
    plt.close(fig)


# ─── Figure 4: LOSO ───────────────────────────────────────────────────────────

def fig_loso(res, out):
    dfs = {k: per_subject(res, f"loso_{k}") for k, _ in MODELS}
    fig = plt.figure(figsize=(183 * MM, 112 * MM))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.05], hspace=0.55, wspace=0.45)

    # (a) balanced accuracy per task, all available subjects per model
    for j, t in enumerate(TASKS):
        ax = fig.add_subplot(gs[0, j])
        data = {k: values(dfs[k], t) for k, _ in MODELS}
        base = values(dfs["none"], t, system="majority").mean()
        lim = (0.15, 0.45) if t == "quad" else (0.30, 0.75)
        model_panel(ax, data, t, base, lim, None)
        if j == 0:
            ax.set_ylabel("Balanced accuracy (window level)")
            ax.text(-0.42, 1.12, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")

    # (b) positive-class F1: model vs always-"high" predictor, EEG-only LOSO
    bottom = gs[1, :].subgridspec(1, 2, width_ratios=[1, 1.45], wspace=0.32)
    ax = fig.add_subplot(bottom[0, 0])
    df = dfs["none"]
    markers = {"val": "o", "aro": "s", "dom": "^", "lik": "D"}
    for t in ["val", "aro", "dom", "lik"]:
        x = values(df, t, system="majority", metric="f1_pos")
        y = values(df, t, metric="f1_pos").reindex(x.index)
        ax.scatter(x, y, s=10, marker=markers[t], facecolors="none",
                   edgecolors=COL["none"], linewidths=0.7, label=TNAME[t])
    ax.plot([0.3, 1.02], [0.3, 1.02], color=GREY_D, lw=0.7, ls="--")
    ax.set_xlim(0.3, 1.02); ax.set_ylim(0.3, 1.02)
    ax.set_xticks(np.arange(0.3, 1.01, 0.1))
    ax.set_xlabel('Positive-class F1 of a constant "high" predictor')
    ax.set_ylabel("Positive-class F1 of EEG-only PGCN")
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.2)
    ax.text(-0.33, 1.06, "b", transform=ax.transAxes, fontsize=9, fontweight="bold")

    # (c) fraction of windows predicted "high", per subject
    ax = fig.add_subplot(bottom[0, 1])
    rng = np.random.default_rng(2)
    for i, t in enumerate(["val", "aro", "dom", "lik"]):
        for k_i, (k, label) in enumerate(MODELS):
            v = values(dfs[k], t, metric="pred_pos").values
            x0 = i + (k_i - 1) * 0.25
            ax.scatter(np.full(len(v), x0) + rng.uniform(-0.07, 0.07, len(v)), v, s=5,
                       color=COL[k], alpha=0.45, linewidths=0)
            ax.plot([x0 - 0.1, x0 + 0.1], [v.mean()] * 2, color=COL[k], lw=1.4)
    ax.set_xticks(range(4)); ax.set_xticklabels([TNAME[t] for t in ["val", "aro", "dom", "lik"]])
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel('Fraction of windows predicted "high"')
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.5); ax.set_axisbelow(True)
    ax.text(-0.21, 1.06, "c", transform=ax.transAxes, fontsize=9, fontweight="bold")

    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.savefig(out / "fig4_loso.pdf", bbox_inches="tight")
    plt.close(fig)


# ─── Figure 5: test-time video manipulation ───────────────────────────────────

def fig_shuffle(res, out):
    sh = pd.read_csv(res / "shuffle" / "video_shuffle_per_subject.csv")
    eeg = per_subject(res, "within_none")
    conds = [("original", "Own trial video", "o", "filled"),
             ("other_trial", "Another trial's video", "s", "none"),
             ("subject_mean", "Participant-mean video", "^", "none")]
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 62 * MM), sharey=True)
    for ax, fus in zip(axes, ["early", "late"]):
        d = sh[sh.fusion == fus]
        for i, t in enumerate(TASKS):
            col = f"{t}_bacc_window"
            ok = values(eeg, t, subjects=FUSION_SUBJECTS).index      # complete test sets
            ref = values(eeg, t, subjects=FUSION_SUBJECTS).mean()
            ax.plot([i - 0.32, i + 0.32], [ref, ref], color=COL["none"], lw=1.2)
            for j, (c, label, mk, fill) in enumerate(conds):
                v = d[(d.condition == c) & d.subject.isin(ok)][col].values
                lo, hi = boot_ci(v)
                colour = COL[fus] if c == "original" else (GREY_D if c == "other_trial" else GREY_L)
                ax.errorbar(i + (j - 1) * 0.2, v.mean(),
                            yerr=[[v.mean() - lo], [hi - v.mean()]], fmt=mk, ms=3.8,
                            color=colour, mfc=colour if fill == "filled" else "white",
                            mec=colour, mew=0.8, elinewidth=0.9, capsize=0)
        ax.set_xticks(range(5)); ax.set_xticklabels([TNAME[t] for t in TASKS])
        style_axis(ax, (0.15, 0.90))
        ax.set_title("Early fusion" if fus == "early" else "Late fusion", color=INK)
    axes[0].set_ylabel("Balanced accuracy (window level)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=GREY_D, mfc=GREY_D, ms=4, label="Own trial video (model colour)"),
         Line2D([], [], marker="s", ls="", color=GREY_D, mfc="white", ms=4, label="Another trial's video"),
         Line2D([], [], marker="^", ls="", color=GREY_L, mfc="white", ms=4, label="Participant-mean video"),
         Line2D([], [], color=COL["none"], lw=1.2, label="EEG-only (same participants)")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "fig5_shuffle.pdf", bbox_inches="tight")
    plt.close(fig)


# ─── Figure 6: embedding probes ───────────────────────────────────────────────

def fig_probes(res, out):
    idn = pd.read_csv(res / "probes" / "P1_identity.csv")
    aff = pd.read_csv(res / "probes" / "P2_P3_affect_probes.csv")
    VC = {"raw": "#0072B2", "centred": "#D55E00"}
    fig = plt.figure(figsize=(183 * MM, 60 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 2, 2], wspace=0.5)

    ax = fig.add_subplot(gs[0, 0])
    for i, v in enumerate(["raw", "centred"]):
        acc = idn[idn.version == v].identity_acc.iloc[0]
        ax.bar(i, acc, width=0.6, color=VC[v])
        ax.text(i, acc + 0.02, f"{acc:.3f}", ha="center", va="bottom", fontsize=6.5)
    ax.axhline(idn.chance.iloc[0], color=GREY_D, ls="--", lw=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Raw", "Centred"])
    ax.set_ylim(0, 1.1); ax.set_ylabel("Identity accuracy (22 classes)")
    ax.set_title("Identity", color=INK)
    ax.text(-0.75, 1.06, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")

    for p_i, (probe, title, letter) in enumerate(
            [("P2_seen_subject", "Seen participants, unseen trials", "b"),
             ("P3_unseen_subject", "Unseen participants (LOSO)", "c")]):
        ax = fig.add_subplot(gs[0, p_i + 1])
        tasks = ["val", "aro", "quad"]
        for i, t in enumerate(tasks):
            for j, v in enumerate(["raw", "centred"]):
                r = aff[(aff.probe == probe) & (aff.task == t) & (aff.version == v)].iloc[0]
                x = i + (j - 0.5) * 0.32
                ax.bar(x, r.bacc_pooled, width=0.3, color=VC[v])
                ax.plot([x - 0.15, x + 0.15], [r.perm_null_95] * 2, color=INK, lw=0.9)
            ax.plot([i - 0.35, i + 0.35], [0.25 if t == "quad" else 0.5] * 2,
                    color=GREY_D, ls="--", lw=0.8)
        ax.set_xticks(range(3)); ax.set_xticklabels([TNAME[t] for t in tasks])
        ax.set_ylim(0, 0.9)
        ax.set_ylabel("Balanced accuracy (pooled)") if p_i == 0 else None
        ax.set_title(title, color=INK)
        ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.5); ax.set_axisbelow(True)
        ax.text(-0.18, 1.06, letter, transform=ax.transAxes, fontsize=9, fontweight="bold")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    h = [Patch(color=VC["raw"], label="Raw embedding"),
         Patch(color=VC["centred"], label="Participant-mean centred"),
         Line2D([], [], color=INK, lw=0.9, label="95th percentile of permutation null"),
         Line2D([], [], color=GREY_D, ls="--", lw=0.8, label="Chance")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(out / "fig6_probes.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True, help="unzipped revision_results folder")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    res, out = Path(a.res), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_within(res, out); print("fig3_within.pdf")
    fig_loso(res, out); print("fig4_loso.pdf")
    fig_shuffle(res, out); print("fig5_shuffle.pdf")
    fig_probes(res, out); print("fig6_probes.pdf")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

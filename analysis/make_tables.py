"""
make_tables.py
==============
Writes every results table of the manuscript and the Supplementary Information
as LaTeX, directly from the CSV files in revision_results.zip, so that no number
is typed by hand.

  tables/table1_main.tex          balanced accuracy, main configurations
  tables/table2_fusion_tests.tex  fusion vs EEG-only, matched participants
  tables/tabS_macro_f1.tex        macro-F1, main configurations
  tables/tabS_posf1.tex           positive-class F1 vs a constant "high" predictor
  tables/tabS_ablation.tex        ablation, balanced accuracy
  tables/tabS_trial.tex           trial-level (mean-probability) balanced accuracy
  tables/tabS_probes.tex          ViT embedding probes
  tables/tabS_shuffle.tex         test-time video manipulation
  tables/tabS_drift.tex           session-drift check
  tables/tabS_allparticipants.tex balanced accuracy with all participants (sensitivity)

Absent classes: unless --all_participants is given, a participant-task pair is
dropped when that participant's test set lacks one of the task's classes
(n_true_classes column), because balanced accuracy is then averaged over the
classes present only. Statistics are read from stats_complete/ (produced by
stats_analysis.py --complete_classes); the sensitivity table uses stats_all/.

Usage:
  python make_tables.py --res revision_results --out tables
"""

import argparse
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd

TASKS = ["val", "aro", "dom", "lik", "quad"]
BIN = ["val", "aro", "dom", "lik"]
TN = {"val": "Val", "aro": "Aro", "dom": "Dom", "lik": "Lik", "quad": "Quad"}
TNF = {"val": "Valence", "aro": "Arousal", "dom": "Dominance", "lik": "Liking", "quad": "Quadrant"}
FUS = [f"s{i:02d}" for i in range(1, 23)]
NCLS = {"val": 2, "aro": 2, "dom": 2, "lik": 2, "quad": 4}
COMPLETE = True          # set False by --all_participants
STATS = "stats_complete"


def r3(x, nd=3):
    """Round half up after removing float noise, so 0.7355 -> 0.736 everywhere."""
    q = Decimal(1).scaleb(-nd)
    v = Decimal(repr(round(float(x), 6))).quantize(q, rounding=ROUND_HALF_UP)
    return str(abs(v) if v == 0 else v)          # no "-0.000"
ABL = [("none", "Full PGCN"), ("flat_gcn", "A1 Flat GCN"),
       ("no_band_attn", "A2 No band attention"), ("struct_only", "A3 Structural edges only"),
       ("no_hjorth", "A4 No Hjorth features")]


def ps(res, cfg):
    return pd.read_csv(res / "reeval_bs" / f"{cfg}_per_subject.csv")


def col(df, task, system="model", metric="bal_acc", level="window", subjects=None,
        complete=None):
    complete = COMPLETE if complete is None else complete
    d = df[(df.level == level) & (df.task == task) & (df.system == system)]
    if complete:
        d = d[d.n_true_classes >= NCLS[task]]
    s = d.set_index("subject")[metric]
    if subjects is not None:
        s = s.reindex([x for x in subjects if x in s.index])
    return s


def ms(s, nd=3):
    return f"${r3(s.mean(), nd)}\\pm{r3(s.std(ddof=1), nd)}$"


def mark(p):
    if pd.isna(p) or p >= .05:
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*"


def fmtp(p):
    return "$<0.001$" if p < 0.001 else f"${p:.3f}$"


def write(out, name, body):
    (out / name).write_text(body)
    print("wrote", name)


def abl_cfg(mode, abl):
    if abl == "none":
        return f"{mode}_none"
    if mode == "loso" and abl == "no_band_attn":
        return "loso_no_band"
    return f"{mode}_{abl}"


# ─── Table 1 ──────────────────────────────────────────────────────────────────

def excluded_note(res):
    """Text listing how many participants each task loses to absent classes."""
    parts = []
    for mode, lab in [("within", "within-participant"), ("loso", "leave-one-subject-out")]:
        d = ps(res, f"{mode}_none")
        d = d[(d.level == "window") & (d.system == "model")]
        bits = []
        for t in TASKS:
            bad = d[(d.task == t) & (d.n_true_classes < NCLS[t])].subject
            if len(bad):
                n22 = sum(x in FUS for x in bad)
                bits.append(f"{TNF[t].lower()} {len(bad)} of 32 ({n22} of 22)")
        if bits:
            parts.append(f"{lab}: " + ", ".join(bits))
    return "; ".join(parts)


def table1_body(res, stats, complete):
    f1 = pd.read_csv(res / stats / "F1_vs_majority_window_bal_acc.csv").set_index(["config", "task"])
    f2 = pd.read_csv(res / stats / "F2_fusion_vs_eeg_window_bal_acc.csv")
    lines = []
    for mode, label in [("within", "Within-participant"), ("loso", "Leave-one-subject-out")]:
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{label}}}}}\\\\")
        eeg, early = ps(res, f"{mode}_none"), ps(res, f"{mode}_early")
        if complete:   # baseline is identical for both subsets once classes are complete
            rows = [("Majority baseline", eeg, "majority", None, None),
                    ("EEG-only ($n{=}32$)", eeg, "model", None, f"{mode}_none"),
                    ("EEG-only ($n{=}22$)", eeg, "model", FUS, None)]
        else:
            rows = [("Majority ($n{=}32$)", eeg, "majority", None, None),
                    ("EEG-only ($n{=}32$)", eeg, "model", None, f"{mode}_none"),
                    ("Majority ($n{=}22$)", early, "majority", FUS, None),
                    ("EEG-only ($n{=}22$)", eeg, "model", FUS, None)]
        rows += [
            ("Early fusion ($n{=}22$)", early, "model", FUS, f"{mode}_early"),
            ("Late fusion ($n{=}22$)", ps(res, f"{mode}_late"), "model", FUS, f"{mode}_late"),
        ]
        for name, df, system, subs, cfg in rows:
            cells = []
            for t in TASKS:
                x = col(df, t, system, subjects=subs, complete=complete)
                c = f"{r3(x.mean())}\\pm{r3(x.std(ddof=1))}"
                sup = ""
                if cfg is not None and (cfg, t) in f1.index:
                    sup += mark(f1.loc[(cfg, t)].p_bh)
                if cfg is not None and cfg.endswith(("early", "late")):
                    r = f2[(f2.a == cfg) & (f2.b == f"{mode}_none") & (f2.task == t)]
                    if len(r) and r.p_bh.iloc[0] < .05:
                        sup += "\\dagger"
                cells.append(f"${c}^{{{sup}}}$" if sup else f"${c}$")
            lines.append(f"{name} & " + " & ".join(cells) + "\\\\")
        lines.append("\\midrule" if mode == "within" else "")
    return "\n".join(l for l in lines if l)


TAB_HEAD = r"""\footnotesize
\setlength{\tabcolsep}{2pt}
\begin{tabular}{lccccc}
\toprule
Configuration & Valence & Arousal & Dominance & Liking & Quadrant \\
\midrule
"""


def table1(res, out):
    body = r"""\begin{table}[t]
\centering
\caption{Balanced accuracy (mean $\pm$ SD over participants, window level) under trial-level partitioning on DEAP. A participant is included for a task only if its test set contains every class of that task (excluded: """ + excluded_note(res) + r"""); in these sets any predictor that ignores its input has an expected balanced accuracy of 0.50 (binary tasks) or 0.25 (quadrant), and the majority-class baseline, which predicts for every window the most frequent class among the training trials the model saw, scores exactly these values. Supplementary Table~S9 gives the same analysis with all participants. Asterisks: model differs from the majority baseline (paired Wilcoxon signed-rank test over participants, Benjamini-Hochberg corrected across all 14 configurations and 5 tasks; $^{*}p_{\mathrm{BH}}<0.05$, $^{**}p_{\mathrm{BH}}<0.01$). $^{\dagger}$Fusion differs from EEG-only on the same participants ($p_{\mathrm{BH}}<0.05$; Table~\ref{tab:fusion}).}
\label{tab:main}
""" + TAB_HEAD + table1_body(res, STATS, True) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write(out, "table1_main.tex", body)


def tab_allparticipants(res, out):
    body = r"""\begin{table}[h]
\centering
\caption{Sensitivity analysis: balanced accuracy (mean $\pm$ SD over participants, window level) with all participants included, as in Table~1 of the main text but without excluding participants whose test set lacks a class. For such a participant, balanced accuracy averages recall over the classes present only, so a constant predictor can score 0, 1 or $1/k$ for $k$ present classes; this is why the majority-baseline means differ from 0.50 and 0.25. Markers and corrections as in Table~1, computed on this sample.}
\label{tab:allpart}
""" + TAB_HEAD + table1_body(res, "stats_all", False) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write(out, "tabS_allparticipants.tex", body)


# ─── Table 2 ──────────────────────────────────────────────────────────────────

def table2(res, out):
    f2 = pd.read_csv(res / STATS / "F2_fusion_vs_eeg_window_bal_acc.csv")
    rows = []
    for t in TASKS:
        for fus, name in [("early", "Early"), ("late", "Late")]:
            cells = [TNF[t] if fus == "early" else "", name]
            for mode in ("within", "loso"):
                r = f2[(f2.a == f"{mode}_{fus}") & (f2.b == f"{mode}_none") & (f2.task == t)].iloc[0]
                cells += [f"${int(r.n)}$",
                          f"${r3(r.mean_diff)}$ (${r3(r.ci_lo)}$, ${r3(r.ci_hi)}$)",
                          f"${r.rank_biserial:.2f}$", fmtp(r.p_bh) + ("" if r.p_bh >= .05 else "$^{*}$")]
            rows.append(" & ".join(cells) + "\\\\")
    body = r"""\begin{table}[t]
\centering
\caption{Fusion minus EEG-only balanced accuracy on the same participants (window level). $n$: participants with every class present in the test set (of 22). $\Delta$: mean paired difference with 95\% bootstrap confidence interval over participants; $r_{rb}$: matched-pairs rank-biserial correlation; $p_{\mathrm{BH}}$: two-sided Wilcoxon signed-rank test, Benjamini-Hochberg corrected across the fusion family (both protocols and the early-versus-late comparison). $^{*}p_{\mathrm{BH}}<0.05$.}
\label{tab:fusion}
\footnotesize
\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{llcccccccc}
\toprule
 & & \multicolumn{4}{c}{Within-participant} & \multicolumn{4}{c}{Leave-one-subject-out} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}
Task & Fusion & $n$ & $\Delta$ (95\% CI) & $r_{rb}$ & $p_{\mathrm{BH}}$ & $n$ & $\Delta$ (95\% CI) & $r_{rb}$ & $p_{\mathrm{BH}}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write(out, "table2_fusion_tests.tex", body)


# ─── Supplementary tables ─────────────────────────────────────────────────────

def tab_metric(res, out, metric, fname, caption, label, level="window"):
    lines = []
    for mode, lab in [("within", "Within-participant"), ("loso", "Leave-one-subject-out")]:
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{lab}}}}}\\\\")
        eeg, early, late = ps(res, f"{mode}_none"), ps(res, f"{mode}_early"), ps(res, f"{mode}_late")
        for name, df, system, subs in [
                ("Majority baseline ($n{=}32$)", eeg, "majority", None),
                ("EEG-only PGCN ($n{=}32$)", eeg, "model", None),
                ("Majority baseline ($n{=}22$)", early, "majority", FUS),
                ("EEG-only PGCN ($n{=}22$)", eeg, "model", FUS),
                ("Early fusion ($n{=}22$)", early, "model", FUS),
                ("Late fusion ($n{=}22$)", late, "model", FUS)]:
            cells = [ms(col(df, t, system, metric, level, subs)) for t in TASKS]
            lines.append(f"{name} & " + " & ".join(cells) + "\\\\")
        if mode == "within":
            lines.append("\\midrule")
    body = f"""\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3.5pt}}
\\begin{{tabular}}{{lccccc}}
\\toprule
Configuration & Valence & Arousal & Dominance & Liking & Quadrant \\\\
\\midrule
""" + "\n".join(lines) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    write(out, fname, body)


def tab_posf1(res, out):
    lines = []
    for mode, lab in [("within", "Within-participant"), ("loso", "Leave-one-subject-out")]:
        lines.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{lab}}}}}\\\\")
        for fus, name, subs in [("none", "EEG-only ($n{=}32$)", None),
                                ("early", "Early fusion ($n{=}22$)", FUS),
                                ("late", "Late fusion ($n{=}22$)", FUS)]:
            df = ps(res, f"{mode}_{fus}")
            cells = []
            for t in BIN:
                m = col(df, t, "model", "f1_pos", subjects=subs).mean()
                c = col(df, t, "majority", "f1_pos", subjects=subs).mean()
                cells.append(f"${r3(m)}$ / ${r3(c)}$")
            pp = [f"${r3(col(df, t, 'model', 'pred_pos', subjects=subs).mean(), 2)}$" for t in BIN]
            lines.append(f"{name} & " + " & ".join(cells) + " & " + " & ".join(pp) + "\\\\")
        if mode == "within":
            lines.append("\\midrule")
    body = r"""\begin{table}[h]
\centering
\caption{Positive-class F1 (the metric used for checkpoint selection) for each model and for the majority-class baseline, and the mean fraction of windows each model predicted as ``high''. Under leave-one-subject-out evaluation the majority baseline predicts ``high'' for every window, so its positive-class F1 equals $2p/(1+p)$, where $p$ is the proportion of high-rated trials.}
\label{tab:posf1}
\footnotesize
\setlength{\tabcolsep}{1.6pt}
\begin{tabular}{lcccccccc}
\toprule
 & \multicolumn{4}{c}{Positive-class F1: model / majority baseline} & \multicolumn{4}{c}{Fraction predicted ``high''} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Model & Val & Aro & Dom & Lik & Val & Aro & Dom & Lik \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write(out, "tabS_posf1.tex", body)


def tab_ablation(res, out):
    f3 = pd.read_csv(res / STATS / "F3_ablation_window_bal_acc.csv")
    lines = []
    for name_i, (abl, name) in enumerate(ABL):
        cells = []
        for mode in ("within", "loso"):
            df = ps(res, abl_cfg(mode, abl))
            for t in TASKS:
                cells.append(f"${r3(col(df, t).mean())}$")
        lines.append(f"{name} & " + " & ".join(cells) + "\\\\")
    minp = f3.p_bh.min()
    body = r"""\begin{table}[h]
\centering
\caption{Ablation study: mean balanced accuracy (window level) of the full PGCN and four variants, within-participant and leave-one-subject-out ($n{=}32$, minus participants whose test set lacks a class: within-participant dominance 28, liking 28, quadrant 19; leave-one-subject-out dominance and quadrant 31). None of the 40 variant-versus-full comparisons is significant after Benjamini-Hochberg correction (smallest $p_{\mathrm{BH}}=""" + f"{minp:.2f}" + r"""$).}
\label{tab:ablation}
\footnotesize
\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{lcccccccccc}
\toprule
 & \multicolumn{5}{c}{Within-participant} & \multicolumn{5}{c}{Leave-one-subject-out} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-11}
Model & Val & Aro & Dom & Lik & Quad & Val & Aro & Dom & Lik & Quad \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write(out, "tabS_ablation.tex", body)


def tab_probes(res, out):
    idn = pd.read_csv(res / "probes" / "P1_identity.csv")
    aff = pd.read_csv(res / "probes" / "P2_P3_affect_probes.csv")
    lines = []
    for probe, lab in [("P2_seen_subject", "Seen participants, unseen trials"),
                       ("P3_unseen_subject", "Unseen participants (leave-one-subject-out)")]:
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{lab}}}}}\\\\")
        for t in TASKS:
            for v in ("raw", "centred"):
                r = aff[(aff.probe == probe) & (aff.task == t) & (aff.version == v)].iloc[0]
                perm = ("--" if pd.isna(r.get("perm_p")) else
                        f"${r3(r.perm_null_95)}$ & ${r3(r.perm_p)}$")
                if perm == "--":
                    perm = "-- & --"
                lines.append(f"{TNF[t] if v == 'raw' else ''} & {v.capitalize()} & "
                             f"${r3(r.bacc_pooled)}$ & "
                             f"${r3(r.majority_bacc_pooled)}$ & {perm}\\\\")
        if probe.startswith("P2"):
            lines.append("\\midrule")
    raw = idn[idn.version == "raw"].identity_acc.iloc[0]
    cen = idn[idn.version == "centred"].identity_acc.iloc[0]
    body = r"""\begin{table}[h]
\centering
\caption{Logistic-regression probes on the frozen ViT-B/16 trial embeddings of the 22 participants with facial video. Identity accuracy (22 classes, five-fold cross-validation): """ + f"{raw:.3f}" + r""" for raw embeddings and """ + f"{cen:.3f}" + r""" after subtracting each participant's mean training-trial embedding (chance 0.045). Affect probes report balanced accuracy pooled over the test trials of all participants (so every class is present), the majority-class baseline, the 95th percentile of a label-permutation null (labels shuffled within each training participant, 100 permutations) and the permutation $p$-value (minimum attainable 0.010).}
\label{tab:probes}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llcccc}
\toprule
Task & Embedding & Balanced accuracy & Majority & Null 95th pct. & $p_{\mathrm{perm}}$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write(out, "tabS_probes.tex", body)


def tab_shuffle(res, out):
    tt = pd.read_csv(res / "shuffle" / "video_shuffle_tests_complete.csv")
    tt = tt[tt.level == "window"]
    lines = []
    for fus in ("early", "late"):
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{'Early' if fus == 'early' else 'Late'} fusion}}}}\\\\")
        for t in TASKS:
            o = tt[(tt.fusion == fus) & (tt.task == t) & (tt.manipulation == "other_trial")].iloc[0]
            m = tt[(tt.fusion == fus) & (tt.task == t) & (tt.manipulation == "subject_mean")].iloc[0]
            lines.append(f"{TNF[t]} & ${int(o.n)}$ & ${r3(o.original)}$ & ${r3(o.manipulated)}$ & {fmtp(o.p_bh)} & "
                         f"${r3(m.manipulated)}$ & {fmtp(m.p_bh)}\\\\")
        if fus == "early":
            lines.append("\\midrule")
    body = r"""\begin{table}[h]
\centering
\caption{Test-time video manipulation on the trained within-participant fusion models (window-level balanced accuracy; $n$: participants of 22 whose test set contains every class of the task). The EEG input is unchanged; each test trial receives either the embedding of a randomly chosen different trial of the same participant (20 repetitions, averaged) or the participant's mean training-trial embedding. $p_{\mathrm{BH}}$: Wilcoxon signed-rank test against the original, Benjamini-Hochberg corrected over all 40 manipulation tests (window and trial level).}
\label{tab:shuffle}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccc}
\toprule
Task & $n$ & Original & Another trial's video & $p_{\mathrm{BH}}$ & Participant-mean video & $p_{\mathrm{BH}}$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write(out, "tabS_shuffle.tex", body)


def tab_drift(res, out):
    d = pd.read_csv(res / "probes" / "drift_check.csv")
    above = int((d.rho_embedding_predicts_position > d.rho_null_95).sum())
    rows = [("Embedding predicts position (cross-validated ridge)",
             d.rho_embedding_predicts_position.mean(), d.rho_null_mean.mean()),
            ("Valence label vs. presentation position", d.rho_val_label_vs_position.mean(), np.nan),
            ("Arousal label vs. presentation position", d.rho_aro_label_vs_position.mean(), np.nan),
            ("Dominance label vs. presentation position", d.rho_dom_label_vs_position.mean(), np.nan),
            ("Liking label vs. presentation position", d.rho_lik_label_vs_position.mean(), np.nan)]
    lines = [f"{n} & ${v:+.3f}$ & {'--' if np.isnan(r) else f'${r:+.3f}$'}\\\\" for n, v, r in rows]
    body = r"""\begin{table}[h]
\centering
\caption{Session-drift check on the 22 participants with facial video (Spearman correlations, mean over participants). Presentation position is the order in which each participant viewed the 40 clips (official DEAP ratings file). Cross-validated regression is biased below zero under the null, so the embedding row is compared with the same statistic computed after shuffling positions (20 shuffles per participant). Embeddings exceeded the 95th percentile of their own null for """ + f"{above} of {len(d)}" + r""" participants.}
\label{tab:drift}
\footnotesize
\begin{tabular}{lcc}
\toprule
Quantity & Mean $\rho$ & Shuffled-position reference \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write(out, "tabS_drift.tex", body)


def tab_trial(res, out):
    tab_metric(res, out, "bal_acc", "tabS_trial.tex",
               "Trial-level balanced accuracy (mean $\\pm$ SD over participants). Each trial's prediction is the class with the highest mean probability over its 29 windows. Within-participant values rest on 6 test trials per participant and are therefore coarse.",
               "tab:trial", level="trial_mean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", default="tables")
    ap.add_argument("--all_participants", action="store_true",
                    help="keep participants whose test set lacks a class (sensitivity)")
    a = ap.parse_args()
    global COMPLETE, STATS
    if a.all_participants:
        COMPLETE, STATS = False, "stats_all"
    res, out = Path(a.res), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table1(res, out)
    table2(res, out)
    tab_metric(res, out, "macro_f1", "tabS_macro_f1.tex",
               "Macro-F1 (mean $\\pm$ SD over participants, window level) for the main configurations and the majority-class baseline. A constant predictor obtains a low macro-F1 (about 0.35 to 0.40 on the binary tasks), so models can exceed the baseline on this metric without carrying information about the labels; balanced accuracy (Table~1) is the primary metric.",
               "tab:macrof1")
    tab_posf1(res, out)
    tab_ablation(res, out)
    tab_trial(res, out)
    tab_probes(res, out)
    tab_shuffle(res, out)
    tab_drift(res, out)
    tab_allparticipants(res, out)


if __name__ == "__main__":
    main()

"""
pgcn_statistics_plots.py  —  Stats + publication-quality plots
==============================================================
"""
import os, json, argparse, warnings
from pathlib import Path
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family":"DejaVu Serif","font.size":11,"axes.titlesize":12,
    "axes.labelsize":11,"xtick.labelsize":10,"ytick.labelsize":10,
    "legend.fontsize":10,"figure.dpi":150,"savefig.dpi":300,
    "savefig.bbox":"tight","axes.spines.top":False,"axes.spines.right":False,
})
PALETTE={"EEG-Only":"#2C7BB6","Early Fusion":"#D7191C","Late Fusion":"#1A9641",
         "Flat GCN":"#FDAE61","No Band Attn":"#ABD9E9",
         "Struct Only":"#A6D96A","No Hjorth":"#F4A582","Full PGCN":"#2C7BB6"}
TASKS=["val","aro","dom","lik","quad"]
TASK_LABELS=["Valence","Arousal","Dominance","Liking","Quadrant"]

# ─── load ─────────────────────────────────────────────────────────────────────

def load_exp(root,exp):
    csv=Path(root)/exp/"aggregate"/"per_subject.csv"
    if csv.exists(): return pd.read_csv(csv,index_col=0)
    rows={}
    for sd in sorted(Path(root).glob(f"{exp}/s*")):
        mp=sd/"metrics.json"
        if mp.exists(): rows[sd.name]=json.load(open(mp))
    for fd in sorted(Path(root).glob(f"{exp}/fold_s*")):
        mp=fd/"metrics.json"
        if mp.exists(): rows[fd.name.replace("fold_","")]=json.load(open(mp))
    return pd.DataFrame(rows).T if rows else None

def load_all(res_eeg,res_fusion):
    exps={
        "EEG-Only (Within)":     (res_eeg,   "within_none"),
        "EEG-Only (LOSO)":       (res_eeg,   "loso_none"),
        "Early Fusion (Within)": (res_fusion, "within_early"),
        "Early Fusion (LOSO)":   (res_fusion, "loso_early"),
        "Late Fusion (Within)":  (res_fusion, "within_late"),
        "Late Fusion (LOSO)":    (res_fusion, "loso_late"),
        "Flat GCN":              (res_eeg,    "within_none_abl_flat_gcn"),
        "No Band Attn":          (res_eeg,    "within_none_abl_no_band_attn"),
        "Struct Only":           (res_eeg,    "within_none_abl_struct_only"),
        "No Hjorth":             (res_eeg,    "within_none_abl_no_hjorth"),
    }
    data={}
    for label,(root,exp) in exps.items():
        if root is None: continue
        df=load_exp(root,exp)
        if df is not None:
            data[label]=df
            print(f"  ✔ {label:30s}  ({len(df)} subjects)")
    return data

# ─── statistics ───────────────────────────────────────────────────────────────

def wilcoxon(a,b):
    if np.all(a==b): return 0.,1.,"ns"
    try: stat,p=stats.wilcoxon(a,b,alternative="two-sided",zero_method="wilcox")
    except: return 0.,1.,"ns"
    stars="***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"
    return float(stat),float(p),stars

def run_stats(data,out_dir,mode="within"):
    sfx=f"({'Within' if mode=='within' else 'LOSO'})"
    mdls=[k for k in [f"EEG-Only {sfx}",f"Early Fusion {sfx}",f"Late Fusion {sfx}"]
          if k in data]
    rows=[]
    print(f"\n{'─'*70}\n  WILCOXON TESTS  {sfx}\n{'─'*70}")
    for t,tl in zip(TASKS,TASK_LABELS):
        col=f"{t}_f1"
        for m1,m2 in combinations(mdls,2):
            if col not in data[m1].columns or col not in data[m2].columns: continue
            a=data[m1][col].dropna().values; b=data[m2][col].dropna().values
            n=min(len(a),len(b)); a,b=a[:n],b[:n]
            stat,p,stars=wilcoxon(a,b)
            r=float(np.sqrt(stat/(n*(n+1)/2+1e-8)))
            d=a.mean()-b.mean()
            rows.append({"Task":tl,"A":m1,"B":m2,"MeanA":f"{a.mean():.4f}",
                         "MeanB":f"{b.mean():.4f}","Δ":f"{d:+.4f}",
                         "p":f"{p:.4f}","Sig":stars,"r":f"{r:.3f}","n":n})
            print(f"  {tl:12s}  {m1:25s} vs {m2:25s}  Δ={d:+.4f}  p={p:.4f}  {stars}")
    if rows:
        pd.DataFrame(rows).to_csv(Path(out_dir)/f"wilcoxon_{mode}.csv",index=False)

# ─── bar chart ────────────────────────────────────────────────────────────────

def plot_bars(data,out_dir,mode="within"):
    sfx=f"({'Within' if mode=='within' else 'LOSO'})"
    mdls={"EEG-Only":f"EEG-Only {sfx}",
          "Early Fusion":f"Early Fusion {sfx}",
          "Late Fusion":f"Late Fusion {sfx}"}
    pres={k:v for k,v in mdls.items() if v in data}
    if not pres: return
    fig,ax=plt.subplots(figsize=(10,4.5))
    x=np.arange(len(TASKS)); n_m=len(pres); w=.22
    offs=np.linspace(-(n_m-1)*w/2,(n_m-1)*w/2,n_m)
    for off,(lbl,ek) in zip(offs,pres.items()):
        df=data[ek]
        ms=[df[f"{t}_f1"].mean() if f"{t}_f1" in df.columns else 0 for t in TASKS]
        ss=[df[f"{t}_f1"].std()  if f"{t}_f1" in df.columns else 0 for t in TASKS]
        bars=ax.bar(x+off,ms,w,yerr=ss,label=lbl,color=PALETTE.get(lbl,"#888"),
                    alpha=.85,error_kw=dict(elinewidth=1.2,capsize=3),
                    edgecolor="white",linewidth=.5)
        for bar,m in zip(bars,ms):
            if m>0:
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.012,
                        f"{m:.2f}",ha="center",va="bottom",fontsize=7.5,
                        fontweight="bold",color=PALETTE.get(lbl,"#333"))
    ax.set_xlabel("Emotional Dimension"); ax.set_ylabel("Macro-F1 Score")
    ax.set_title(f"Performance — {'Within-Subject' if mode=='within' else 'LOSO'}")
    ax.set_xticks(x); ax.set_xticklabels(TASK_LABELS)
    ax.set_ylim(0,1.15)
    ax.axhline(.5,ls="--",color="#aaa",lw=1.,label="Chance (0.5)")
    ax.legend(loc="upper right",framealpha=.9); ax.grid(axis="y",alpha=.3)
    for fmt in ["pdf","png"]:
        fig.savefig(Path(out_dir)/f"bar_{mode}.{fmt}")
    plt.close(); print(f"  ✔ bar_{mode}.pdf/png")

# ─── boxplots ─────────────────────────────────────────────────────────────────

def plot_boxes(data,out_dir,mode="within"):
    sfx=f"({'Within' if mode=='within' else 'LOSO'})"
    mdls={"EEG-Only":f"EEG-Only {sfx}","Early Fusion":f"Early Fusion {sfx}",
          "Late Fusion":f"Late Fusion {sfx}"}
    pres={k:v for k,v in mdls.items() if v in data}
    if not pres: return
    fig,axes=plt.subplots(1,len(TASKS),figsize=(14,4),sharey=False)
    for ax,t,tl in zip(axes,TASKS,TASK_LABELS):
        col=f"{t}_f1"; rows=[]
        for lbl,ek in pres.items():
            if col not in data[ek].columns: continue
            for v in data[ek][col].dropna():
                rows.append({"Model":lbl,"F1":float(v)})
        if not rows: ax.axis("off"); continue
        df=pd.DataFrame(rows)
        sns.boxplot(data=df,x="Model",y="F1",ax=ax,width=.5,linewidth=1.1,
                    palette={k:PALETTE.get(k,"#888") for k in pres},
                    flierprops=dict(marker="o",markersize=3,alpha=.4))
        sns.stripplot(data=df,x="Model",y="F1",ax=ax,size=3.5,alpha=.35,
                      jitter=True,palette={k:PALETTE.get(k,"#888") for k in pres})
        ax.set_xlabel(""); ax.set_title(tl)
        ax.set_ylabel("Macro-F1" if t==TASKS[0] else "")
        ax.axhline(.5,ls="--",color="#aaa",lw=1.)
        ax.tick_params(axis="x",rotation=20)
    fig.suptitle(f"Per-Subject F1 — {'Within' if mode=='within' else 'LOSO'}",y=1.01)
    fig.tight_layout()
    for fmt in ["pdf","png"]: fig.savefig(Path(out_dir)/f"boxplot_{mode}.{fmt}")
    plt.close(); print(f"  ✔ boxplot_{mode}.pdf/png")

# ─── gap ──────────────────────────────────────────────────────────────────────

def plot_gap(data,out_dir):
    mdls=["EEG-Only","Early Fusion","Late Fusion"]
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,t in zip(axes,["val","aro","quad"]):
        col=f"{t}_f1"; x=np.arange(len(mdls)); w=.3
        def g(k): return data[k][col].mean() if k in data and col in data[k].columns else 0
        def s(k): return data[k][col].std()  if k in data and col in data[k].columns else 0
        wm=[g(f"{m} (Within)") for m in mdls]; lm=[g(f"{m} (LOSO)") for m in mdls]
        ws=[s(f"{m} (Within)") for m in mdls]; ls=[s(f"{m} (LOSO)") for m in mdls]
        ax.bar(x-w/2,wm,w,yerr=ws,label="Within",color="#2C7BB6",alpha=.85,
               error_kw=dict(elinewidth=1,capsize=3),edgecolor="white")
        ax.bar(x+w/2,lm,w,yerr=ls,label="LOSO",  color="#D7191C",alpha=.85,
               error_kw=dict(elinewidth=1,capsize=3),edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(mdls,rotation=15,ha="right")
        ax.set_ylabel("Macro-F1"); ax.set_ylim(0,1.05)
        ax.set_title(TASK_LABELS[TASKS.index(t)])
        ax.axhline(.5,ls="--",color="#aaa",lw=1.)
        if t=="val": ax.legend()
    fig.suptitle("Within-Subject vs LOSO Generalization Gap",fontsize=13)
    fig.tight_layout()
    for fmt in ["pdf","png"]: fig.savefig(Path(out_dir)/f"gap.{fmt}")
    plt.close(); print(f"  ✔ gap.pdf/png")

# ─── ablation ─────────────────────────────────────────────────────────────────

def plot_ablation(data,out_dir):
    order={"Full PGCN":"EEG-Only (Within)","Flat GCN":"Flat GCN",
           "No Band Attn":"No Band Attn","Struct Only":"Struct Only",
           "No Hjorth":"No Hjorth"}
    pres={k:v for k,v in order.items() if v in data}
    if len(pres)<2: return
    fig,axes=plt.subplots(1,len(TASKS),figsize=(14,5))
    for ax,t,tl in zip(axes,TASKS,TASK_LABELS):
        col=f"{t}_f1"; lbls=list(pres.keys())
        ms=[data[v][col].mean() if col in data[v].columns else 0 for v in pres.values()]
        ss=[data[v][col].std()  if col in data[v].columns else 0 for v in pres.values()]
        y=np.arange(len(lbls))
        ax.barh(y,ms,xerr=ss,color=[PALETTE.get(l,"#888") for l in lbls],
                alpha=.85,edgecolor="white",error_kw=dict(elinewidth=1.2,capsize=3))
        full=ms[0] if ms else 0
        for i,(m,s_) in enumerate(zip(ms,ss)):
            drop=full-m
            lbl=f"{m:.3f}"+(f" (−{drop:.3f})" if i>0 and drop>0 else "")
            ax.text(m+max(s_,.01),y[i],lbl,va="center",fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(lbls if t==TASKS[0] else [""]*len(lbls))
        ax.set_xlabel("Macro-F1"); ax.set_title(tl)
        ax.axvline(.5,ls="--",color="#aaa",lw=1.); ax.set_xlim(0,1.15)
    fig.suptitle("Ablation Study — Within-Subject",fontsize=13)
    fig.tight_layout()
    for fmt in ["pdf","png"]: fig.savefig(Path(out_dir)/f"ablation.{fmt}")
    plt.close(); print(f"  ✔ ablation.pdf/png")

# ─── latex table ──────────────────────────────────────────────────────────────

def latex_table(data,out_dir):
    mdls=[("EEG-Only","EEG-Only (Within)","EEG-Only (LOSO)"),
          ("Early Fusion","Early Fusion (Within)","Early Fusion (LOSO)"),
          ("Late Fusion","Late Fusion (Within)","Late Fusion (LOSO)")]
    bw={t:0. for t in TASKS}; bl={t:0. for t in TASKS}
    for _,wk,lk in mdls:
        for t in TASKS:
            col=f"{t}_f1"
            if wk in data and col in data[wk].columns:
                bw[t]=max(bw[t],data[wk][col].mean())
            if lk in data and col in data[lk].columns:
                bl[t]=max(bl[t],data[lk][col].mean())
    def cell(df,t,best):
        col=f"{t}_f1"
        if df is None or col not in df.columns: return "---"
        m=df[col].mean(); s=df[col].std()
        c=f"{m:.3f}$\\pm${s:.3f}"
        return r"\textbf{"+c+"}" if abs(m-best)<1e-4 else c
    lines=[r"\begin{table*}[h]",r"\centering",
           r"\caption{Macro-F1 (Mean$\,\pm\,$SD) on DEAP. "
           r"Trial-level evaluation. Best per column in \textbf{bold}.}",
           r"\label{tab:main}",r"\begin{tabular}{l"+"c"*10+"}",r"\toprule",
           r" & \multicolumn{5}{c}{Within-Subject} & \multicolumn{5}{c}{LOSO} \\",
           r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
           "Model & "+" & ".join(TASK_LABELS*2)+r" \\",r"\midrule"]
    for ml,wk,lk in mdls:
        cw=" & ".join(cell(data.get(wk),t,bw[t]) for t in TASKS)
        cl=" & ".join(cell(data.get(lk),t,bl[t]) for t in TASKS)
        lines.append(f"{ml} & {cw} & {cl} "+r"\\")
    lines+=[r"\bottomrule",r"\end{tabular}",r"\end{table*}"]
    (Path(out_dir)/"results_table.tex").write_text("\n".join(lines))
    print(f"  ✔ results_table.tex")

# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results_eeg",   default="/kaggle/working/results_eeg")
    p.add_argument("--results_fusion",default="/kaggle/working/results_fusion")
    p.add_argument("--out_dir",       default="/kaggle/working/figures")
    a=p.parse_args()
    Path(a.out_dir).mkdir(parents=True,exist_ok=True)
    print("Loading results …")
    data=load_all(a.results_eeg,a.results_fusion)
    if not data: print("No data found."); return
    for mode in ["within","loso"]:
        run_stats(data,a.out_dir,mode)
        plot_bars(data,a.out_dir,mode)
        plot_boxes(data,a.out_dir,mode)
    plot_gap(data,a.out_dir)
    plot_ablation(data,a.out_dir)
    latex_table(data,a.out_dir)
    print(f"\n✔ All outputs → {a.out_dir}")

if __name__=="__main__":
    main()

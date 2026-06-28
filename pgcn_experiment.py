"""
pgcn_experiment.py
==================
Complete experiment runner with all fixes incorporated.
Use this file for ALL 14 experiments instead of pgcn_train.py.

Fixes included:
  1. Smaller model (hidden=64) to prevent overfitting on 28 training trials
  2. Cross-entropy loss instead of focal (more stable for small datasets)
  3. Window-level evaluation (174 test windows vs 6 trial-level predictions)
  4. Mean F1 across tasks for early stopping (not loss)
  5. Summary reads from per_subject dict (no yt_all concatenation error)
  6. Works for within-subject, LOSO, early fusion, late fusion, all ablations

Usage in Kaggle notebook:
    import sys; sys.path.insert(0, "/kaggle/working")
    from pgcn_experiment import run_experiment, print_summary

    store = run_experiment(
        data_dir = "/kaggle/input/datasets/jerinanan/pgcn-deap-preprocessed",
        out_dir  = "/kaggle/working/results_eeg/within_none",
        mode     = "within",   # "within" or "loso"
        fusion   = "none",     # "none", "early", "late"
        ablation = "none",     # "none", "flat_gcn", "no_band_attn",
                               # "struct_only", "no_hjorth"
        force    = False,      # True to rerun completed subjects
    )
    print_summary(store, out_dir="/kaggle/working/results_eeg/within_none/aggregate")
"""

import gc, json, time
import numpy as np
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.loader import DataLoader as GeoLoader

# ─── import project modules ───────────────────────────────────────────────────
import pgcn_model as M
from pgcn_core import DEAPDataset, make_scheduler, EEGAugment, TASKS

# ─── hyperparameters ──────────────────────────────────────────────────────────
CFGS = {
    ("within", "none"):  dict(lr=3e-4, wd=1e-3, bs=32, ep=200, pat=30, wu=10, clip=1.),
    ("within", "early"): dict(lr=1e-4, wd=1e-3, bs=32, ep=200, pat=30, wu=10, clip=1.),
    ("within", "late"):  dict(lr=1e-4, wd=1e-3, bs=32, ep=200, pat=30, wu=10, clip=1.),
    ("loso",   "none"):  dict(lr=3e-4, wd=1e-3, bs=16, ep=100, pat=15, wu=10, clip=1.),
    ("loso",   "early"): dict(lr=1e-4, wd=1e-3, bs=16, ep=100, pat=15, wu=10, clip=1.),
    ("loso",   "late"):  dict(lr=1e-4, wd=1e-3, bs=16, ep=100, pat=15, wu=10, clip=1.),
}


# ─── model builder ────────────────────────────────────────────────────────────

def build_model(mode="within", fusion="none", ablation="none"):
    """
    Smaller model for within-subject (hidden=64),
    standard size for LOSO (hidden=128).
    """
    abl  = M.ABLATION_CONFIGS.get(ablation, M.ABLATION_CONFIGS["none"])
    loso = (mode == "loso")
    return M.PyramidalGCN(
        in_dim         = abl["in_dim"],
        hidden         = 128 if loso else 64,
        heads          = 4,
        dropout        = 0.50 if loso else 0.45,
        use_band_attn  = abl["use_band_attn"],
        use_hierarchy  = abl["use_hierarchy"],
        use_functional = abl["use_functional"],
        n_domains      = 32 if (loso and fusion == "none") else 0,
        lambda_grl     = 0.3,
        fusion         = fusion,
        edge_dropout   = 0.10 if loso else 0.0,
        feat_dropout   = 0.10 if loso else 0.0,
    )


# ─── loss function ────────────────────────────────────────────────────────────

class MultiTaskCELoss(nn.Module):
    """Simple cross-entropy multi-task loss. Stable for small datasets."""
    def forward(self, out: dict, y: torch.Tensor):
        if y.dim() == 1:
            y = y.unsqueeze(0)
        if y.dim() == 2 and y.size(1) != 5:
            y = y.view(out["val"].size(0), 5)
        total = 0.
        for i, t in enumerate(["val", "aro", "dom", "lik"]):
            total = total + F.cross_entropy(out[t], y[:, i].long())
        total = total + 2. * F.cross_entropy(out["quad"], y[:, 4].long())
        return total, {"total": total}


# ─── evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Window-level evaluation. Returns dict of acc and F1 per task."""
    model.eval()
    yt = {t: [] for t in TASKS}
    yp = {t: [] for t in TASKS}

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            out = model(batch)
        y = batch.y
        if y.dim() == 1:
            y = y.unsqueeze(0)
        yn = y.cpu().numpy()
        for i, t in enumerate(["val", "aro", "dom", "lik"]):
            yt[t].append(yn[:, i])
            yp[t].append(out[t].argmax(1).cpu().numpy())
        yt["quad"].append(yn[:, 4])
        yp["quad"].append(out["quad"].argmax(1).cpu().numpy())

    res = {}
    for t in ["val", "aro", "dom", "lik"]:
        a = np.concatenate(yt[t])
        b = np.concatenate(yp[t])
        res[f"{t}_f1"]  = float(f1_score(a, b, average="binary", zero_division=0))
        res[f"{t}_acc"] = float(accuracy_score(a, b))
    a = np.concatenate(yt["quad"])
    b = np.concatenate(yp["quad"])
    res["quad_f1"]  = float(f1_score(a, b, average="macro", zero_division=0))
    res["quad_acc"] = float(accuracy_score(a, b))
    return res


def mean_binary_f1(metrics: dict) -> float:
    """Mean F1 across the 4 binary tasks — used for early stopping."""
    return float(np.mean([metrics[f"{t}_f1"]
                           for t in ["val", "aro", "dom", "lik"]]))


# ─── subject helpers ──────────────────────────────────────────────────────────

def get_available_subjects(data_dir: str) -> list:
    all_s = [f"s{i:02d}" for i in range(1, 33)]
    return [s for s in all_s
            if (Path(data_dir) / s / "meta.json").exists()]


def get_real_video_subjects(data_dir: str) -> list:
    real = []
    for s in get_available_subjects(data_dir):
        mp = Path(data_dir) / s / "meta.json"
        m  = json.load(open(mp))
        if m.get("has_real_video", False):
            real.append(s)
    if not real:
        print("  [WARN] has_real_video flag missing — using all subjects")
        return get_available_subjects(data_dir)
    return real


def get_subjects(data_dir: str, fusion: str) -> list:
    if fusion in ("early", "late"):
        return get_real_video_subjects(data_dir)
    return get_available_subjects(data_dir)


# ─── results store ────────────────────────────────────────────────────────────

class ResultsStore:
    """Stores per-subject metrics. Summary reads from per_subject dict."""

    def __init__(self):
        self.per_subject = {}

    def add(self, subj: str, metrics: dict):
        self.per_subject[subj] = metrics


def print_summary(store: ResultsStore,
                  out_dir: Optional[str] = None) -> dict:
    """
    Compute and print mean ± std across all subjects.
    Saves per_subject.csv and global_metrics.json if out_dir given.
    Returns global metrics dict.
    """
    import pandas as pd

    results = store.per_subject
    if not results:
        print("  [WARN] No results to summarise.")
        return {}

    print("\n" + "=" * 60)
    print(f"  RESULTS  ({len(results)} subjects)")
    print("=" * 60)

    global_m = {}
    for task, label in [("val_f1",  "Valence F1"),
                         ("aro_f1",  "Arousal F1"),
                         ("dom_f1",  "Dominance F1"),
                         ("lik_f1",  "Liking F1"),
                         ("quad_f1", "Quadrant F1"),
                         ("val_acc",  "Valence Acc"),
                         ("aro_acc",  "Arousal Acc"),
                         ("dom_acc",  "Dominance Acc"),
                         ("lik_acc",  "Liking Acc"),
                         ("quad_acc", "Quadrant Acc")]:
        vals = [v[task] for v in results.values() if task in v]
        if not vals:
            continue
        mn, sd = np.mean(vals), np.std(vals)
        global_m[f"{task}_mean"] = float(mn)
        global_m[f"{task}_std"]  = float(sd)
        if task.endswith("_f1"):
            print(f"  {label:15s}: {mn:.4f} ± {sd:.4f}  "
                  f"(min={np.min(vals):.4f}  max={np.max(vals):.4f})")

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).T.to_csv(
            f"{out_dir}/per_subject.csv", float_format="%.4f")
        json.dump(global_m,
                  open(f"{out_dir}/global_metrics.json", "w"), indent=2)
        print(f"\n  ✔ Saved → {out_dir}/")

    return global_m


# ─── training loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion,
                    device, scaler, clip, augment):
    model.train()
    total = 0.; nb = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        if augment is not None:
            batch.x = augment(batch.x)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            out  = model(batch)
            loss, _ = criterion(out, batch.y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item(); nb += 1
    return total / max(1, nb)


# ─── within-subject ───────────────────────────────────────────────────────────

def _run_within(data_dir, out_dir, fusion, ablation,
                store, device, force, log_every=20):
    cfg       = CFGS[("within", fusion)]
    no_hjorth = (ablation == "no_hjorth")
    augment   = EEGAugment(noise_std=.03, ch_drop_p=.05, feat_mask_p=.05)
    subjects  = get_subjects(data_dir, fusion)
    crit      = MultiTaskCELoss()

    print(f"\n  within  fusion={fusion}  ablation={ablation}"
          f"  ({len(subjects)} subjects)")

    for subj in subjects:
        sdir  = Path(out_dir) / subj
        ckpt  = sdir / "best.pt"
        rpath = sdir / "metrics.json"

        if rpath.exists() and not force:
            print(f"  [SKIP] {subj}")
            store.add(subj, json.load(open(rpath)))
            continue

        sdir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'─'*55}\n  {subj}")

        ds_tr = DEAPDataset(data_dir, [subj], "train", no_hjorth)
        ds_vl = DEAPDataset(data_dir, [subj], "val",   no_hjorth)
        ds_te = DEAPDataset(data_dir, [subj], "test",  no_hjorth)

        if len(ds_tr) == 0:
            print("  [SKIP] no data"); continue

        kw = dict(batch_size=cfg["bs"], num_workers=2, pin_memory=True)
        dl_tr = GeoLoader(ds_tr, shuffle=True,  **kw)
        dl_vl = GeoLoader(ds_vl, shuffle=False, **kw)
        dl_te = GeoLoader(ds_te, shuffle=False, **kw)

        model  = build_model("within", fusion, ablation).to(device)
        opt    = optim.AdamW(model.parameters(),
                              lr=cfg["lr"], weight_decay=cfg["wd"])
        sched  = make_scheduler(opt, cfg["wu"], cfg["ep"])
        scaler = torch.amp.GradScaler("cuda")

        best_f1 = -1.; wait = 0

        for ep in range(1, cfg["ep"] + 1):
            tl = train_one_epoch(model, dl_tr, opt, crit,
                                  device, scaler, cfg["clip"], augment)
            vm = evaluate(model, dl_vl, device)
            mf = mean_binary_f1(vm)
            sched.step()

            if ep % log_every == 0 or ep == 1:
                print(f"  ep{ep:3d}/{cfg['ep']}  tr={tl:.3f}  "
                      f"val={vm['val_f1']:.3f}  aro={vm['aro_f1']:.3f}  "
                      f"dom={vm['dom_f1']:.3f}  lik={vm['lik_f1']:.3f}  "
                      f"quad={vm['quad_f1']:.3f}  mean={mf:.3f}")

            if mf > best_f1 + 1e-4:
                best_f1 = mf; wait = 0
                torch.save(model.state_dict(), ckpt)
            else:
                wait += 1
                if wait >= cfg["pat"]:
                    print(f"  Early stop ep{ep}  best={best_f1:.4f}")
                    break

        model.load_state_dict(
            torch.load(ckpt, map_location=device, weights_only=True))
        tm = evaluate(model, dl_te, device)
        store.add(subj, tm)
        json.dump(tm, open(rpath, "w"), indent=2)
        print(f"  TEST  val={tm['val_f1']:.4f}  aro={tm['aro_f1']:.4f}  "
              f"dom={tm['dom_f1']:.4f}  lik={tm['lik_f1']:.4f}  "
              f"quad={tm['quad_f1']:.4f}")

        del model, opt, sched, scaler
        torch.cuda.empty_cache(); gc.collect()


# ─── LOSO ────────────────────────────────────────────────────────────────────

def _run_loso(data_dir, out_dir, fusion, ablation,
              store, device, force, log_every=10):
    cfg       = CFGS[("loso", fusion)]
    no_hjorth = (ablation == "no_hjorth")
    augment   = EEGAugment(noise_std=.03, ch_drop_p=.05, feat_mask_p=.05)
    all_avail = get_available_subjects(data_dir)
    test_subs = get_subjects(data_dir, fusion)
    crit      = MultiTaskCELoss()

    print(f"\n  loso  fusion={fusion}  ablation={ablation}"
          f"  ({len(test_subs)} test folds)")

    for test_s in test_subs:
        fdir  = Path(out_dir) / f"fold_{test_s}"
        ckpt  = fdir / "best.pt"
        rpath = fdir / "metrics.json"

        if rpath.exists() and not force:
            print(f"  [SKIP] fold {test_s}")
            store.add(test_s, json.load(open(rpath)))
            continue

        fdir.mkdir(parents=True, exist_ok=True)
        train_s = [s for s in all_avail if s != test_s]
        print(f"\n{'─'*55}\n  fold test={test_s}  train={len(train_s)}")

        ds_tr = DEAPDataset(data_dir, train_s,  "train", no_hjorth)
        ds_vl = DEAPDataset(data_dir, train_s,  "val",   no_hjorth)
        ds_te = DEAPDataset(data_dir, [test_s], "all",   no_hjorth)

        if len(ds_tr) == 0:
            print("  [SKIP] no data"); continue

        kw = dict(batch_size=cfg["bs"], num_workers=2, pin_memory=True)
        dl_tr = GeoLoader(ds_tr, shuffle=True,  **kw)
        dl_vl = GeoLoader(ds_vl, shuffle=False, **kw)
        dl_te = GeoLoader(ds_te, shuffle=False, **kw)

        model  = build_model("loso", fusion, ablation).to(device)
        opt    = optim.Adam(model.parameters(),
                             lr=cfg["lr"], weight_decay=cfg["wd"])
        sched  = make_scheduler(opt, cfg["wu"], cfg["ep"])
        scaler = torch.amp.GradScaler("cuda")

        best_f1 = -1.; wait = 0

        for ep in range(1, cfg["ep"] + 1):
            tl = train_one_epoch(model, dl_tr, opt, crit,
                                  device, scaler, cfg["clip"], augment)
            vm = evaluate(model, dl_vl, device)
            mf = mean_binary_f1(vm)
            sched.step()

            if ep % log_every == 0 or ep == 1:
                print(f"  ep{ep:3d}/{cfg['ep']}  tr={tl:.3f}  "
                      f"val={vm['val_f1']:.3f}  aro={vm['aro_f1']:.3f}  "
                      f"mean={mf:.3f}")

            if mf > best_f1 + 1e-4:
                best_f1 = mf; wait = 0
                torch.save(model.state_dict(), ckpt)
            else:
                wait += 1
                if wait >= cfg["pat"]:
                    print(f"  Early stop ep{ep}  best={best_f1:.4f}")
                    break

        model.load_state_dict(
            torch.load(ckpt, map_location=device, weights_only=True))
        tm = evaluate(model, dl_te, device)
        store.add(test_s, tm)
        json.dump(tm, open(rpath, "w"), indent=2)
        print(f"  TEST  val={tm['val_f1']:.4f}  aro={tm['aro_f1']:.4f}  "
              f"dom={tm['dom_f1']:.4f}  lik={tm['lik_f1']:.4f}  "
              f"quad={tm['quad_f1']:.4f}")

        del model, opt, sched, scaler, ds_tr, ds_vl, ds_te
        del dl_tr, dl_vl, dl_te
        torch.cuda.empty_cache(); gc.collect()


# ─── public API ───────────────────────────────────────────────────────────────

def run_experiment(data_dir: str,
                   out_dir:  str,
                   mode:     str = "within",
                   fusion:   str = "none",
                   ablation: str = "none",
                   device:   str = "cuda",
                   force:    bool = False) -> ResultsStore:
    """
    Run one experiment and return a ResultsStore with per-subject metrics.

    Parameters
    ----------
    data_dir : path to preprocessed dataset (preprocessed_eeg or preprocessed_fusion)
    out_dir  : where to save checkpoints and metrics.json files
    mode     : "within" or "loso"
    fusion   : "none", "early", "late"
    ablation : "none", "flat_gcn", "no_band_attn", "struct_only", "no_hjorth"
    device   : "cuda" or "cpu"
    force    : if True, rerun even if metrics.json already exists
    """
    assert mode    in ("within", "loso"),            f"Unknown mode: {mode}"
    assert fusion  in ("none", "early", "late"),     f"Unknown fusion: {fusion}"
    assert ablation in M.ABLATION_CONFIGS,           f"Unknown ablation: {ablation}"

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    dev   = torch.device(device)
    store = ResultsStore()

    print("=" * 60)
    print(f"  mode={mode}  fusion={fusion}  ablation={ablation}")
    print(f"  data_dir : {data_dir}")
    print(f"  out_dir  : {out_dir}")
    print("=" * 60)

    if mode == "within":
        _run_within(data_dir, out_dir, fusion, ablation,
                    store, dev, force)
    else:
        _run_loso(data_dir, out_dir, fusion, ablation,
                  store, dev, force)

    return store

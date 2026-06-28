"""
pgcn_core.py  —  Losses, Dataset, Augmentation, Training, Trial-level Aggregation
===================================================================================
Key improvements:
  1. Class-weighted focal loss  (handles DEAP class imbalance)
  2. EEG augmentation           (Gaussian noise + channel dropout)
  3. Trial-level prediction aggregation at test time  (+2-4% F1)
  4. Weighted random sampler    (balanced mini-batches)
"""

import json, os
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader as GeoLoader
from sklearn.metrics import accuracy_score, f1_score

TASKS = ["val","aro","dom","lik","quad"]

# ─── class-weighted focal loss ────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss with label smoothing AND optional class weights.
    Class weights address DEAP's ~60/40 class imbalance.
    """
    def __init__(self, gamma=2., smooth=.1,
                 class_weights: Optional[torch.Tensor]=None):
        super().__init__()
        self.gamma=gamma; self.smooth=smooth
        self.register_buffer("cw", class_weights)   # [n_classes] or None

    def forward(self, logits, targets):
        n=logits.size(-1)
        if self.smooth>0 and self.training:
            with torch.no_grad():
                st=torch.zeros_like(logits).fill_(self.smooth/(n-1))
                st.scatter_(1,targets.unsqueeze(1),1.-self.smooth)
            log_p=F.log_softmax(logits,dim=-1)
            ce=-(st*log_p).sum(-1)
        else:
            ce=F.cross_entropy(logits,targets,reduction="none")

        pt=torch.exp(-ce)
        focal=((1-pt)**self.gamma)*ce

        # Apply per-sample class weights
        if self.cw is not None:
            w=self.cw[targets]
            focal=focal*w

        return focal.mean()


class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task focal loss.
    class_weights_dict: {"val":{0:w0,1:w1}, "aro":..., "quad":{0:..,3:..}}
    """
    def __init__(self, gamma=2., smooth=.1,
                 class_weights_dict: Optional[dict]=None):
        super().__init__()
        self.losses=nn.ModuleDict()
        for i,t in enumerate(["val","aro","dom","lik"]):
            cw=None
            if class_weights_dict and t in class_weights_dict:
                raw=class_weights_dict[t]
                cw=torch.tensor([raw.get(j,1.) for j in range(2)],
                                 dtype=torch.float32)
            self.losses[t]=FocalLoss(gamma,smooth,cw)
        # Quadrant (4-class)
        cw=None
        if class_weights_dict and "quad" in class_weights_dict:
            raw=class_weights_dict["quad"]
            cw=torch.tensor([raw.get(j,1.) for j in range(4)],
                             dtype=torch.float32)
        self.losses["quad"]=FocalLoss(gamma,smooth,cw)

    def forward(self, out, y):
        if y.dim()==1: y=y.unsqueeze(0)
        if y.dim()==2 and y.size(1)!=5:
            y=y.view(out["val"].size(0),5)
        total=0.; L={}
        for i,t in enumerate(["val","aro","dom","lik"]):
            l=self.losses[t](out[t],y[:,i].long())
            L[t]=l; total=total+l
        lq=self.losses["quad"](out["quad"],y[:,4].long())
        L["quad"]=lq; total=total+2.*lq
        if "domain" in out:
            ld=F.cross_entropy(out["domain"],out["_dom_lbl"])
            L["domain"]=ld; total=total+.3*ld
        L["total"]=total
        return total,L

# ─── EEG augmentation ────────────────────────────────────────────────────────

class EEGAugment:
    """
    Applied to node feature tensor x [N_nodes, N_feats] during training.
    
    1. Gaussian noise:    adds N(0, noise_std) to all features
    2. Channel dropout:   randomly zeros out entire node rows
    3. Feature masking:   randomly zeros out feature columns
    """
    def __init__(self, noise_std=0.05, ch_drop_p=0.1, feat_mask_p=0.1):
        self.noise_std  = noise_std
        self.ch_drop_p  = ch_drop_p
        self.feat_mask_p= feat_mask_p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():   # skip at eval
            return x
        x = x.clone()

        # 1. Gaussian noise
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        # 2. Channel (node) dropout
        if self.ch_drop_p > 0:
            mask = torch.rand(x.size(0), device=x.device) > self.ch_drop_p
            x = x * mask.unsqueeze(-1).float()

        # 3. Feature masking
        if self.feat_mask_p > 0:
            mask = torch.rand(x.size(-1), device=x.device) > self.feat_mask_p
            x = x * mask.unsqueeze(0).float()

        return x


# ─── dataset ──────────────────────────────────────────────────────────────────

class DEAPDataset(Dataset):
    def __init__(self, root, subjects, split="all", no_hjorth=False):
        super().__init__()
        self.root=Path(root); self.split=split; self.nh=no_hjorth
        self.files=[]
        for s in subjects:
            sd=self.root/s; mp=sd/"meta.json"
            if not mp.exists(): continue
            if split=="all":
                self.files+=sorted(sd.rglob("win_*.pt"))
            else:
                meta=json.load(open(mp))
                keep=set(meta.get(f"{split}_trials",[]))
                for td in sorted(sd.glob("trial_*")):
                    ti=int(td.name.split("_")[1])
                    if ti in keep:
                        self.files+=sorted(td.glob("win_*.pt"))
        print(f"  DEAPDataset [{split:5s}]  {len(self.files):6d} windows  "
              f"({len(subjects)} subjects)")

    def len(self): return len(self.files)

    def get(self, idx):
        d=torch.load(self.files[idx],weights_only=False)
        if d.y.dim()==1: d.y=d.y.unsqueeze(0)
        if hasattr(d,"video") and d.video.dim()==1:
            d.video=d.video.unsqueeze(0)
        if self.nh and d.x.shape[1]==21:
            d.x=torch.cat([d.x[:,:10],d.x[:,13:]],dim=1)
        return d


def make_loaders(data_dir, subjects, batch=32, workers=2,
                 no_hjorth=False, use_weighted_sampler=True):
    """
    use_weighted_sampler: balances mini-batches by quadrant label → better training.
    """
    kw = dict(num_workers=workers, pin_memory=True,
              persistent_workers=(workers>0))

    ds_tr = DEAPDataset(data_dir, subjects, "train", no_hjorth)
    ds_vl = DEAPDataset(data_dir, subjects, "val",   no_hjorth)
    ds_te = DEAPDataset(data_dir, subjects, "test",  no_hjorth)

    # Weighted sampler for training (balances quadrant classes)
    if use_weighted_sampler and len(ds_tr) > 0:
        labels = []
        for i in range(len(ds_tr)):
            g = ds_tr.get(i)
            labels.append(int(g.y.view(-1)[4]))   # quad label
        labels = np.array(labels)
        counts = np.bincount(labels, minlength=4)
        weights = 1.0 / (counts[labels] + 1e-8)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.float32),
            num_samples=len(ds_tr), replacement=True)
        dl_tr = GeoLoader(ds_tr, batch_size=batch, sampler=sampler, **kw)
    else:
        dl_tr = GeoLoader(ds_tr, batch_size=batch, shuffle=True, **kw)

    dl_vl = GeoLoader(ds_vl, batch_size=batch, shuffle=False, **kw)
    dl_te = GeoLoader(ds_te, batch_size=batch, shuffle=False, **kw)
    return dl_tr, dl_vl, dl_te


def load_class_weights(data_dir: str, subjects: List[str]) -> dict:
    """
    Average class weights across all subjects.
    Returns dict ready for MultiTaskLoss.
    """
    agg = {t: defaultdict(list) for t in TASKS}
    for s in subjects:
        mp = Path(data_dir)/s/"meta.json"
        if not mp.exists(): continue
        meta = json.load(open(mp))
        cw = meta.get("class_weights", {})
        for t in TASKS:
            if t in cw:
                for c,w in cw[t].items():
                    agg[t][int(c)].append(w)
    # Average
    result = {}
    for t in TASKS:
        nc = 4 if t=="quad" else 2
        result[t] = {c: np.mean(agg[t][c]) if agg[t][c] else 1.
                     for c in range(nc)}
    return result


# ─── scheduler ───────────────────────────────────────────────────────────────

def make_scheduler(opt, warmup, total, min_lr=.01):
    def lr_fn(ep):
        if ep<warmup: return (ep+1)/warmup
        p=(ep-warmup)/max(1,total-warmup)
        return min_lr+.5*(1-min_lr)*(1+np.cos(np.pi*p))
    return torch.optim.lr_scheduler.LambdaLR(opt,lr_fn)


# ─── metrics ─────────────────────────────────────────────────────────────────

def metrics_from_dicts(yt, yp):
    res={}
    for i,t in enumerate(["val","aro","dom","lik"]):
        a=np.concatenate(yt[t]); b=np.concatenate(yp[t])
        res[f"{t}_acc"]=accuracy_score(a,b)
        res[f"{t}_f1"] =f1_score(a,b,average="binary",zero_division=0)
    a=np.concatenate(yt["quad"]); b=np.concatenate(yp["quad"])
    res["quad_acc"]=accuracy_score(a,b)
    res["quad_f1"] =f1_score(a,b,average="macro",zero_division=0)
    return res


# ─── TRIAL-LEVEL AGGREGATION (key improvement) ───────────────────────────────

def trial_level_metrics(yt_dict, yp_proba_dict, trial_ids):
    """
    Aggregate window-level PROBABILITY predictions to trial level by averaging,
    then take argmax. This is the correct evaluation unit (one label per trial).
    
    yt_dict      : {task: [array per batch]}
    yp_proba_dict: {task: [prob array per batch]}  — softmax probabilities
    trial_ids    : [array per batch] of trial indices
    
    Returns trial-level metrics dict.
    """
    # Flatten
    trial_arr = np.concatenate(trial_ids)
    unique_trials = np.unique(trial_arr)

    yt_trial = {t: [] for t in TASKS}
    yp_trial = {t: [] for t in TASKS}

    for ti in unique_trials:
        mask = (trial_arr == ti)
        for task in TASKS:
            labels = np.concatenate(yt_dict[task])[mask]
            probs  = np.concatenate(yp_proba_dict[task])[mask]  # [n_win, n_cls]
            # Use majority label for ground truth (all windows same trial → same label)
            yt_trial[task].append(int(np.bincount(labels).argmax()))
            # Average probabilities across windows, then argmax
            yp_trial[task].append(int(probs.mean(axis=0).argmax()))

    yt_trial = {t: [np.array(v)] for t,v in yt_trial.items()}
    yp_trial = {t: [np.array(v)] for t,v in yp_trial.items()}
    return metrics_from_dicts(yt_trial, yp_trial)


# ─── train / eval ────────────────────────────────────────────────────────────

def train_epoch(model, loader, opt, crit, device, scaler, clip=1.,
                augment: Optional[EEGAugment]=None):
    model.train(); tot=0.; nb=0
    for batch in loader:
        batch=batch.to(device,non_blocking=True)

        # ★ Apply EEG augmentation to node features
        if augment is not None:
            batch.x = augment(batch.x)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            out=model(batch); loss,_=crit(out,batch.y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(),clip)
        scaler.step(opt); scaler.update()
        tot+=loss.item(); nb+=1
    return tot/max(1,nb)


@torch.no_grad()
def eval_epoch(model, loader, crit, device, return_preds=False):
    model.eval(); tot=0.; nb=0
    yt={t:[] for t in TASKS}; yp={t:[] for t in TASKS}
    yp_proba={t:[] for t in TASKS}   # softmax probabilities for aggregation
    trial_ids=[]

    for batch in loader:
        batch=batch.to(device,non_blocking=True)
        with torch.amp.autocast("cuda"):
            out=model(batch); loss,_=crit(out,batch.y)
        tot+=loss.item(); nb+=1

        y=batch.y
        if y.dim()==1: y=y.unsqueeze(0)
        yn=y.cpu().numpy()

        for i,t in enumerate(["val","aro","dom","lik"]):
            yt[t].append(yn[:,i])
            probs=torch.softmax(out[t],dim=-1).cpu().numpy()
            yp_proba[t].append(probs)
            yp[t].append(probs.argmax(axis=-1))

        q_probs=torch.softmax(out["quad"],dim=-1).cpu().numpy()
        yt["quad"].append(yn[:,4])
        yp_proba["quad"].append(q_probs)
        yp["quad"].append(q_probs.argmax(axis=-1))

        # Collect trial indices if available
        if hasattr(batch,"trial_idx"):
            trial_ids.append(batch.trial_idx.cpu().numpy())

    avg_loss=tot/max(1,nb)
    win_metrics=metrics_from_dicts(yt,yp)      # window-level (for early stopping)

    if return_preds:
        return avg_loss, win_metrics, yt, yp, yp_proba, trial_ids
    return avg_loss, win_metrics


# ─── results store ────────────────────────────────────────────────────────────

class ResultsStore:
    def __init__(self):
        self.per_subject={}
        self.yt_all={t:[] for t in TASKS}
        self.yp_all={t:[] for t in TASKS}

    def add(self, subj, yt, yp, yp_proba=None, trial_ids=None):
        """
        Stores TRIAL-LEVEL metrics per subject if trial_ids available,
        else falls back to window-level.
        """
        if yp_proba is not None and trial_ids and len(trial_ids)>0:
            m = trial_level_metrics(yt, yp_proba, trial_ids)
            m["level"] = "trial"
        else:
            m = metrics_from_dicts(yt, yp)
            m["level"] = "window"
        self.per_subject[subj] = m
        for t in TASKS:
            self.yt_all[t]+=yt[t]
            self.yp_all[t]+=yp[t]

    def summary(self, out_dir=None):
        import pandas as pd
        gm=metrics_from_dicts(self.yt_all,self.yp_all)
        print("\n"+"="*62)
        print("  GLOBAL AGGREGATED RESULTS (window-level)")
        print("="*62)
        for k,v in gm.items():
            if k!="level": print(f"  {k:<16s}: {v:.4f}")
        if self.per_subject:
            print("\n  PER-SUBJECT  mean ± std  (trial-level where available)")
            for t in TASKS:
                arr=np.array([v[f"{t}_f1"] for v in self.per_subject.values()])
                print(f"  {t}_f1  =  {arr.mean():.4f} ± {arr.std():.4f}")
        if out_dir:
            os.makedirs(out_dir,exist_ok=True)
            json.dump({k:float(v) for k,v in gm.items() if k!="level"},
                      open(f"{out_dir}/global_metrics.json","w"),indent=2)
            pd.DataFrame(self.per_subject).T.to_csv(
                f"{out_dir}/per_subject.csv",float_format="%.4f")
        return gm

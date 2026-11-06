"""
pgcn_train.py  —  Universal trainer with all optimisations
===========================================================
Uses:
  • Class-weighted focal loss      (from meta.json class_weights)
  • EEG augmentation               (noise + channel dropout)
  • Weighted random sampler        (balanced mini-batches)
  • Trial-level test aggregation   (reports trial-level F1)
  • Real-video subject filtering   (fusion experiments)
  • AMP throughout                 (Kaggle GPU efficiency)
  • Auto-resume                    (skips completed subjects)
"""

import gc, json, time, argparse, warnings
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.optim as optim

from pgcn_model import build_model
from pgcn_core  import (MultiTaskLoss, DEAPDataset, make_loaders,
                         load_class_weights, make_scheduler,
                         train_epoch, eval_epoch,
                         EEGAugment, ResultsStore, TASKS)
from torch_geometric.loader import DataLoader as GeoLoader

warnings.filterwarnings("ignore")

# ─── hyperparameter table ─────────────────────────────────────────────────────
CFGS = {
    ("within","none"):  dict(lr=1e-3,wd=1e-4,bs=32, ep=100,pat=15,wu=5,clip=1.),
    ("within","early"): dict(lr=1e-4,wd=1e-2,bs=48, ep=100,pat=15,wu=5,clip=1.),
    ("within","late"):  dict(lr=1e-4,wd=1e-2,bs=48, ep=100,pat=15,wu=5,clip=1.),
    ("loso",  "none"):  dict(lr=1e-3,wd=1e-4,bs=16, ep=80, pat=10,wu=5,clip=1.),
    ("loso",  "early"): dict(lr=1e-4,wd=1e-2,bs=32, ep=80, pat=10,wu=5,clip=1.),
    ("loso",  "late"):  dict(lr=1e-4,wd=1e-2,bs=32, ep=80, pat=10,wu=5,clip=1.),
}

ALL_SUBJECTS = [f"s{i:02d}" for i in range(1,33)]

# ─── subject helpers ──────────────────────────────────────────────────────────

def get_available(data_dir):
    return [s for s in ALL_SUBJECTS
            if (Path(data_dir)/s/"meta.json").exists()]

def get_real_video(data_dir):
    real=[]
    for s in ALL_SUBJECTS:
        mp=Path(data_dir)/s/"meta.json"
        if not mp.exists(): continue
        m=json.load(open(mp))
        if m.get("has_real_video",False): real.append(s)
    if not real:
        print("  [WARN] has_real_video not found — using all subjects")
        return get_available(data_dir)
    return real

# ─── within-subject ───────────────────────────────────────────────────────────

def run_within(args, store: ResultsStore):
    device=torch.device(args.device)
    cfg=CFGS[("within",args.fusion)]
    no_hjorth=(args.ablation=="no_hjorth")
    augment=EEGAugment(noise_std=.05,ch_drop_p=.10,feat_mask_p=.10)

    subjects=(get_real_video(args.data_dir)
              if args.fusion in ("early","late")
              else get_available(args.data_dir))
    print(f"\n  Within-subject  fusion={args.fusion}  abl={args.ablation}"
          f"  ({len(subjects)} subjects)")

    # Load class weights (averaged across all subjects)
    cw_dict=load_class_weights(args.data_dir, subjects)

    for subj in subjects:
        sdir=Path(args.out_dir)/subj
        ckpt=sdir/"best.pt"; rpath=sdir/"metrics.json"

        if rpath.exists() and not args.force:
            print(f"  [SKIP] {subj}")
            store.per_subject[subj]=json.load(open(rpath)); continue

        sdir.mkdir(parents=True,exist_ok=True)
        print(f"\n{'─'*55}\n  {subj}")

        dl_tr,dl_vl,dl_te=make_loaders(
            args.data_dir,[subj],cfg["bs"],args.workers,no_hjorth,
            use_weighted_sampler=True)
        if len(dl_tr.dataset)==0:
            print("  [SKIP] no data"); continue

        model=build_model(args.mode,args.fusion,args.ablation).to(device)
        crit =MultiTaskLoss(gamma=2.,smooth=.1,class_weights_dict=cw_dict)
        opt  =optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=cfg["wd"])
        sched=make_scheduler(opt,cfg["wu"],cfg["ep"])
        scaler=torch.amp.GradScaler("cuda")

        best_vl=1e9; wait=0
        for ep in range(1,cfg["ep"]+1):
            t0=time.time()
            tl=train_epoch(model,dl_tr,opt,crit,device,scaler,cfg["clip"],augment)
            vl,vm=eval_epoch(model,dl_vl,crit,device)
            sched.step()
            if ep%10==0 or ep==1:
                print(f"  ep{ep:3d}/{cfg['ep']}  tr={tl:.4f}  vl={vl:.4f}  "
                      f"val_f1={vm['val_f1']:.3f}  aro_f1={vm['aro_f1']:.3f}  "
                      f"quad_f1={vm['quad_f1']:.3f}  [{time.time()-t0:.1f}s]")
            if vl<best_vl-1e-4: best_vl=vl; wait=0; torch.save(model.state_dict(),ckpt)
            else:
                wait+=1
                if wait>=cfg["pat"]: print(f"  Early stop ep{ep}"); break

        model.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True))
        _,_,yt,yp,yp_proba,trial_ids=eval_epoch(
            model,dl_te,crit,device,return_preds=True)
        store.add(subj,yt,yp,yp_proba,trial_ids)
        json.dump(store.per_subject[subj],open(rpath,"w"),indent=2)
        m=store.per_subject[subj]
        print(f"  TEST [{m.get('level','?')}]  "
              f"val={m['val_f1']:.4f}  aro={m['aro_f1']:.4f}  "
              f"dom={m['dom_f1']:.4f}  lik={m['lik_f1']:.4f}  quad={m['quad_f1']:.4f}")
        del model,opt,sched,scaler
        torch.cuda.empty_cache(); gc.collect()

# ─── LOSO ─────────────────────────────────────────────────────────────────────

def run_loso(args, store: ResultsStore):
    device=torch.device(args.device)
    cfg=CFGS[("loso",args.fusion)]
    no_hjorth=(args.ablation=="no_hjorth")
    augment=EEGAugment(noise_std=.05,ch_drop_p=.10,feat_mask_p=.10)

    all_avail=get_available(args.data_dir)
    test_subjects=(get_real_video(args.data_dir)
                   if args.fusion in ("early","late")
                   else all_avail)
    print(f"\n  LOSO  fusion={args.fusion}  abl={args.ablation}"
          f"  ({len(test_subjects)} test folds)")

    Path(args.out_dir).mkdir(parents=True,exist_ok=True)

    for test_s in test_subjects:
        fdir=Path(args.out_dir)/f"fold_{test_s}"
        ckpt=fdir/"best.pt"; rpath=fdir/"metrics.json"
        if rpath.exists() and not args.force:
            print(f"  [SKIP] fold {test_s}")
            store.per_subject[test_s]=json.load(open(rpath)); continue

        fdir.mkdir(parents=True,exist_ok=True)
        train_s=[s for s in all_avail if s!=test_s]
        print(f"\n{'─'*55}\n  fold test={test_s}  train={len(train_s)}")

        # Load class weights from training subjects only
        cw_dict=load_class_weights(args.data_dir,train_s)

        ds_tr=DEAPDataset(args.data_dir,train_s,"train",no_hjorth)
        ds_vl=DEAPDataset(args.data_dir,train_s,"val",  no_hjorth)
        ds_te=DEAPDataset(args.data_dir,[test_s],"all", no_hjorth)

        kw=dict(batch_size=cfg["bs"],num_workers=args.workers,
                pin_memory=True,persistent_workers=(args.workers>0))
        dl_tr=GeoLoader(ds_tr,shuffle=True, **kw)
        dl_vl=GeoLoader(ds_vl,shuffle=False,**kw)
        dl_te=GeoLoader(ds_te,shuffle=False,**kw)

        if len(ds_tr)==0: print("  [SKIP] no data"); continue

        model=build_model(args.mode,args.fusion,args.ablation).to(device)
        crit =MultiTaskLoss(gamma=2.,smooth=.1,class_weights_dict=cw_dict)
        opt  =optim.Adam(model.parameters(),lr=cfg["lr"],weight_decay=cfg["wd"])
        sched=make_scheduler(opt,cfg["wu"],cfg["ep"])
        scaler=torch.amp.GradScaler("cuda")

        best_vl=1e9; wait=0
        for ep in range(1,cfg["ep"]+1):
            t0=time.time()
            tl=train_epoch(model,dl_tr,opt,crit,device,scaler,cfg["clip"],augment)
            vl,vm=eval_epoch(model,dl_vl,crit,device)
            sched.step()
            if ep%5==0 or ep==1:
                print(f"  ep{ep:3d}/{cfg['ep']}  tr={tl:.4f}  vl={vl:.4f}  "
                      f"val_f1={vm['val_f1']:.3f}  [{time.time()-t0:.1f}s]")
            if vl<best_vl-1e-4: best_vl=vl; wait=0; torch.save(model.state_dict(),ckpt)
            else:
                wait+=1
                if wait>=cfg["pat"]: print(f"  Early stop ep{ep}"); break

        model.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True))
        _,_,yt,yp,yp_proba,trial_ids=eval_epoch(
            model,dl_te,crit,device,return_preds=True)
        store.add(test_s,yt,yp,yp_proba,trial_ids)
        json.dump(store.per_subject[test_s],open(rpath,"w"),indent=2)
        m=store.per_subject[test_s]
        print(f"  TEST [{m.get('level','?')}]  "
              f"val={m['val_f1']:.4f}  aro={m['aro_f1']:.4f}  "
              f"dom={m['dom_f1']:.4f}  lik={m['lik_f1']:.4f}  quad={m['quad_f1']:.4f}")
        del model,opt,sched,scaler,ds_tr,ds_vl,ds_te,dl_tr,dl_vl,dl_te
        torch.cuda.empty_cache(); gc.collect()

# ─── entry point ──────────────────────────────────────────────────────────────

def get_args():
    p=argparse.ArgumentParser()
    p.add_argument("--data_dir",default="/kaggle/working/preprocessed_eeg")
    p.add_argument("--out_dir", default="/kaggle/working/results")
    p.add_argument("--mode",    default="within",choices=["within","loso"])
    p.add_argument("--fusion",  default="none",  choices=["none","early","late"])
    p.add_argument("--ablation",default="none",
                   choices=["none","flat_gcn","no_band_attn","struct_only","no_hjorth"])
    p.add_argument("--device",  default="cuda")
    p.add_argument("--workers", type=int,default=2)
    p.add_argument("--force",   action="store_true")
    return p.parse_args()

def main():
    args=get_args()
    tag=f"{args.mode}_{args.fusion}"
    if args.ablation!="none": tag+=f"_abl_{args.ablation}"
    args.out_dir=str(Path(args.out_dir)/tag)
    Path(args.out_dir).mkdir(parents=True,exist_ok=True)
    print("="*62)
    print(f"  Experiment : {tag}")
    print(f"  Data dir   : {args.data_dir}")
    print("="*62)
    store=ResultsStore()
    if args.mode=="within": run_within(args,store)
    else:                   run_loso(args,store)
    store.summary(out_dir=str(Path(args.out_dir)/"aggregate"))
    print(f"\n✔ Done.")

if __name__=="__main__":
    main()

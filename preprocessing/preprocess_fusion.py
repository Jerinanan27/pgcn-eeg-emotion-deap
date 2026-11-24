"""
preprocess_fusion.py
Exact preprocessing code that produced the released dataset (extracted verbatim from the
Kaggle notebook fusion-preprocess.ipynb, cell that writes it). Approximate entropy is not computed
(feature 15 = 0) and functional edges are |Pearson| top-12.
"""

import os, gc, json, argparse, warnings
from pathlib import Path
from collections import Counter
import h5py
import numpy as np
import torch
from scipy import signal
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.data import Data
from tqdm import tqdm

warnings.filterwarnings("ignore")

FS=128; N_CH=32; BASELINE=3*FS; TRIAL_USE=33*FS
WIN_SAMP=4*FS; STEP_SAMP=1*FS; N_WIN=29

BANDS={"theta":(4,8),"alpha":(8,13),"beta":(13,30),
       "gamma":(30,45),"hgamma":(45,60)}
BAND_LIST=["theta","alpha","beta","gamma","hgamma"]
EDGE_TOPK=12

SYM_PAIRS=[(0,16),(1,17),(2,19),(3,20),(4,21),(5,22),
           (6,24),(7,25),(8,26),(9,27),(10,28),(11,29),(12,30),(13,31)]
BRAIN_REGIONS=[
    [0,1,2,3],[16,17,19,20],[18,4,5,21,22],
    [6,7,8,9],[23,24,25,26,27],[10,11,12,15],[13,14,28,29,30,31],
]

def bandpass(x,lo,hi,fs=FS,order=4):
    nyq=fs/2.; hi=min(hi,nyq-.5); lo=min(lo,nyq-1.)
    if lo>=hi: return np.zeros_like(x)
    b,a=signal.butter(order,[lo/nyq,hi/nyq],btype="band")
    return signal.filtfilt(b,a,x,axis=-1)

def de(x): return .5*np.log(2*np.pi*np.e*(np.var(x,axis=-1)+1e-8))
def log_var(x): return np.log(np.var(x,axis=-1)+1e-8)
def log_en(x): return np.log(np.sum(x**2,axis=-1)+1e-8)

def hjorth(x):
    act=np.var(x,axis=-1); d1=np.diff(x,axis=-1)
    mob=np.sqrt(np.var(d1,axis=-1)/(act+1e-8)); d2=np.diff(d1,axis=-1)
    cmp=np.sqrt(np.var(d2,axis=-1)/(np.var(d1,axis=-1)+1e-8))/(mob+1e-8)
    return np.log(act+1e-8),mob,cmp

def spec_ent(x,fs=FS):
    _,p=signal.welch(x,fs=fs,nperseg=128)
    p=p/(p.sum(axis=-1,keepdims=True)+1e-8)
    return -np.sum(p*np.log(p+1e-8),axis=-1)

def extract_features(win):
    C=win.shape[0]; F=np.zeros((C,21),dtype=np.float32)
    for b,band in enumerate(BAND_LIST):
        lo,hi=BANDS[band]; filt=bandpass(win,lo,hi)
        F[:,b]=de(filt); F[:,5+b]=log_var(filt)
    a,m,c=hjorth(win)
    F[:,10]=a; F[:,11]=m; F[:,12]=c
    F[:,13]=spec_ent(win); F[:,14]=log_en(win); F[:,15]=0.
    pm={}
    for l,r in SYM_PAIRS: pm[l]=(r,True); pm[r]=(l,False)
    for ch in range(C):
        if ch in pm:
            pt,il=pm[ch]
            F[ch,16:21]=(F[ch,:5]-F[pt,:5]) if il else (F[pt,:5]-F[ch,:5])
    return np.clip(F,-20,20)

def build_struct():
    rows,cols,wts=[],[],[]
    for reg in BRAIN_REGIONS:
        for i in reg:
            for j in reg:
                if i!=j: rows.append(i);cols.append(j);wts.append(1.)
    seen=set(zip(rows,cols))
    for l,r in SYM_PAIRS:
        for s,d in [(l,r),(r,l)]:
            if (s,d) not in seen:
                rows.append(s);cols.append(d);wts.append(.5);seen.add((s,d))
    return (torch.tensor([rows,cols],dtype=torch.long),
            torch.tensor(wts,dtype=torch.float32))

STRUCT_EI,STRUCT_EW=build_struct()

def build_func(win,topk=EDGE_TOPK):
    z=(win-win.mean(axis=1,keepdims=True))/(win.std(axis=1,keepdims=True)+1e-8)
    conn=np.abs((z@z.T)/win.shape[1]).astype(np.float32)
    np.fill_diagonal(conn,0)
    rows,cols,wts=[],[],[]
    for i in range(conn.shape[0]):
        idx=np.argsort(conn[i])[::-1][:topk]
        for j in idx:
            if conn[i,j]>0: rows.append(i);cols.append(j);wts.append(float(conn[i,j]))
    if not rows:
        return torch.empty((2,0),dtype=torch.long),torch.empty((0,),dtype=torch.float32)
    return (torch.tensor([rows,cols],dtype=torch.long),
            torch.tensor(wts,dtype=torch.float32))

def make_labels(lbl):
    v=int(lbl[0]>5);a=int(lbl[1]>5);d=int(lbl[2]>5);l=int(lbl[3]>5)
    return torch.tensor([v,a,d,l,v*2+a],dtype=torch.long)

def trial_split(quad,seed=42):
    n=len(quad); idx=np.arange(n)
    nt=max(1,int(.15*n)); nv=max(1,int(.15*n))
    s1=StratifiedShuffleSplit(1,test_size=nt,random_state=seed)
    try: (rest,te),=s1.split(idx,quad)
    except:
        rng=np.random.default_rng(seed); p=rng.permutation(n)
        te=p[:nt]; rest=p[nt:]
    s2=StratifiedShuffleSplit(1,test_size=nv,random_state=seed+1)
    try: (tr,vl),=s2.split(rest,quad[rest])
    except:
        rng=np.random.default_rng(seed+2); p=rng.permutation(len(rest))
        vl=p[:nv]; tr=p[nv:]
    return rest[tr],rest[vl],te

def compute_norm_stats(feats):
    all_f=np.concatenate(feats,axis=0)
    return (all_f.mean(axis=0,keepdims=True).astype(np.float32),
            (all_f.std(axis=0,keepdims=True)+1e-8).astype(np.float32))

def has_real_video(h5f,subj):
    for tname in h5f[subj].keys():
        vid=np.array(h5f[subj][tname]["video_embedding"])
        if not np.allclose(vid,0.,atol=1e-6): return True
    return False

def process_subject(h5f,subj,out_dir):
    sdir=Path(out_dir)/subj
    if (sdir/"meta.json").exists():
        print(f"  [SKIP] {subj}",flush=True); return
    sdir.mkdir(parents=True,exist_ok=True)
    if subj not in h5f:
        print(f"  [WARN] {subj} not in H5",flush=True); return

    real_vid=has_real_video(h5f,subj)
    print(f"Processing {subj} [{'REAL' if real_vid else 'ZERO'} video]...",flush=True)

    grp=h5f[subj]; trials=sorted(grp.keys()); n_t=len(trials)
    quad=np.array([int(np.array(grp[t]["labels"])[0]>5)*2+
                   int(np.array(grp[t]["labels"])[1]>5)
                   for t in trials],dtype=np.int64)
    tr_t,vl_t,te_t=trial_split(quad)
    sp={**{i:"train" for i in tr_t},**{i:"val" for i in vl_t},
        **{i:"test" for i in te_t}}

    train_feats=[]
    for t_idx in tr_t:
        eeg=np.array(grp[trials[t_idx]]["raw_eeg"],dtype=np.float32)[:N_CH]
        seg=eeg[:,BASELINE:BASELINE+TRIAL_USE]
        for w in range(N_WIN):
            s=w*STEP_SAMP
            train_feats.append(extract_features(seg[:,s:s+WIN_SAMP]))
        del eeg,seg
    norm_mean,norm_std=compute_norm_stats(train_feats)
    del train_feats; gc.collect()
    np.save(sdir/"norm_mean.npy",norm_mean)
    np.save(sdir/"norm_std.npy",norm_std)

    total=0; label_counts={t:Counter() for t in ["val","aro","dom","lik","quad"]}
    for t_idx,tname in enumerate(trials):
        td=sdir/f"trial_{t_idx:02d}"; td.mkdir(exist_ok=True)
        eeg=np.array(grp[tname]["raw_eeg"],dtype=np.float32)[:N_CH]
        vid=np.array(grp[tname]["video_embedding"],dtype=np.float32)
        lbl=np.array(grp[tname]["labels"],dtype=np.float32)
        y=make_labels(lbl)
        v_t=torch.tensor(vid,dtype=torch.float32).unsqueeze(0)
        seg=eeg[:,BASELINE:BASELINE+TRIAL_USE]
        yn=y.numpy()
        for i,tn in enumerate(["val","aro","dom","lik"]):
            label_counts[tn][int(yn[i])]+=N_WIN
        label_counts["quad"][int(yn[4])]+=N_WIN
        for w in range(N_WIN):
            s=w*STEP_SAMP; win=seg[:,s:s+WIN_SAMP]
            ft=(extract_features(win)-norm_mean)/norm_std
            ei_f,ew_f=build_func(win)
            g=Data(x=torch.tensor(ft,dtype=torch.float32),
                   edge_index_struct=STRUCT_EI.clone(),
                   edge_weight_struct=STRUCT_EW.clone(),
                   edge_index_func=ei_f,edge_weight_func=ew_f,
                   video=v_t.clone(),
                   y=y.unsqueeze(0),split=sp[t_idx],
                   trial_idx=t_idx,window_idx=w,subject=subj)
            torch.save(g,td/f"win_{w:02d}.pt")
            total+=1
        del eeg,seg,vid,lbl; gc.collect()

    class_weights={}
    for tn,cnt in label_counts.items():
        nc=4 if tn=="quad" else 2; tot=sum(cnt.values())
        class_weights[tn]={c:tot/(nc*cnt.get(c,1)) for c in range(nc)}

    meta={"subject":subj,"n_trials":n_t,"n_windows":N_WIN,
          "total_windows":total,"train_trials":tr_t.tolist(),
          "val_trials":vl_t.tolist(),"test_trials":te_t.tolist(),
          "quad_per_trial":quad.tolist(),"class_weights":class_weights,
          "has_real_video":bool(real_vid)}
    json.dump(meta,open(sdir/"meta.json","w"),indent=2)
    print(f"  ✔ {subj}: {total} windows  has_real_video={real_vid}",flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--h5_path",default="/kaggle/input/datasets/jerinanan/deap-fused-data/fused_dataset.h5")
    p.add_argument("--out_dir",default="/kaggle/working/preprocessed_fusion")
    p.add_argument("--subjects",default="all")
    a=p.parse_args()
    Path(a.out_dir).mkdir(parents=True,exist_ok=True)
    subs=([f"s{i:02d}" for i in range(1,33)] if a.subjects=="all"
          else [s.strip() for s in a.subjects.split(",")])
    print(f"Processing {len(subs)} subjects...",flush=True)
    real,zero=[],[]
    with h5py.File(a.h5_path,"r") as h5f:
        for s in subs:
            try: process_subject(h5f,s,a.out_dir)
            except Exception as e:
                import traceback; print(f"  ERROR {s}: {e}"); traceback.print_exc()
            mp=Path(a.out_dir)/s/"meta.json"
            if mp.exists():
                m=json.load(open(mp))
                (real if m.get("has_real_video") else zero).append(s)
    print(f"\nDONE",flush=True)
    print(f"Real video subjects ({len(real)}): {real}",flush=True)
    print(f"Zero vector subjects({len(zero)}): {zero}",flush=True)
    json.dump({"real_video":real,"zero_video":zero},
              open(Path(a.out_dir)/"video_subject_list.json","w"),indent=2)

if __name__=="__main__":
    main()

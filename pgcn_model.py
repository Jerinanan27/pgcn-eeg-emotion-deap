"""
pgcn_model.py  —  All model architectures
==========================================
PyramidalGCN with full pyramid + all ablation variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool

DEAP_REGIONS = [
    [0,1,2,3],[16,17,19,20],[18,4,5,21,22],
    [6,7,8,9],[23,24,25,26,27],[10,11,12,15],[13,14,28,29,30,31],
]

ABLATION_CONFIGS = {
    "none":         {"use_band_attn":True,  "use_hierarchy":True,  "use_functional":True,  "in_dim":21},
    "flat_gcn":     {"use_band_attn":True,  "use_hierarchy":False, "use_functional":False, "in_dim":21},
    "no_band_attn": {"use_band_attn":False, "use_hierarchy":True,  "use_functional":True,  "in_dim":21},
    "struct_only":  {"use_band_attn":True,  "use_hierarchy":True,  "use_functional":False, "in_dim":21},
    "no_hjorth":    {"use_band_attn":True,  "use_hierarchy":True,  "use_functional":True,  "in_dim":18},
}

class _GRLFn(Function):
    @staticmethod
    def forward(ctx,x,alpha): ctx.alpha=alpha; return x.clone()
    @staticmethod
    def backward(ctx,g): return -ctx.alpha*g,None

class GRL(nn.Module):
    def __init__(self,alpha=1.): super().__init__(); self.alpha=alpha
    def forward(self,x): return _GRLFn.apply(x,self.alpha)

class BandAttn(nn.Module):
    def __init__(self,n=5):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(n,16),nn.ReLU(),nn.Linear(16,n))
    def forward(self,x):
        de=x[:,:5]; w=torch.sigmoid(self.mlp(de.mean(0,keepdim=True)))
        out=x.clone(); out[:,:5]=de*w; return out

class GCNBlock(nn.Module):
    def __init__(self,in_d,out_d,drop=.3):
        super().__init__()
        self.conv=GCNConv(in_d,out_d,add_self_loops=True,normalize=True)
        self.norm=nn.LayerNorm(out_d); self.drop=nn.Dropout(drop)
        self.res=nn.Linear(in_d,out_d,bias=False) if in_d!=out_d else nn.Identity()
    def forward(self,x,ei,ew=None):
        h=self.drop(F.relu(self.norm(self.conv(x,ei,ew))))
        return h+self.res(x)

class MesoAttn(nn.Module):
    def __init__(self,d,heads=4,drop=.1):
        super().__init__()
        self.d=d; self.h=heads; self.hd=d//heads; self.scale=self.hd**-.5
        self.Wq=nn.Linear(d,d,bias=False); self.Wk=nn.Linear(d,d,bias=False)
        self.Wv=nn.Linear(d,d,bias=False); self.Wo=nn.Linear(d,d,bias=False)
        self.norm=nn.LayerNorm(d); self.drop=nn.Dropout(drop)
    def forward(self,x,batch):
        B=int(batch.max())+1; N=32; D=self.d
        h=x.view(B,N,D); out=torch.zeros_like(h)
        for reg in DEAP_REGIONS:
            hr=h[:,reg,:]; nr=len(reg)
            Q=self.Wq(hr).view(B,nr,self.h,self.hd).transpose(1,2)
            K=self.Wk(hr).view(B,nr,self.h,self.hd).transpose(1,2)
            V=self.Wv(hr).view(B,nr,self.h,self.hd).transpose(1,2)
            A=self.drop(torch.softmax(Q@K.transpose(-2,-1)*self.scale,dim=-1))
            ro=(A@V).transpose(1,2).contiguous().view(B,nr,D)
            for i,n in enumerate(reg): out[:,n,:]=self.Wo(ro)[:,i,:]
        return self.norm(out+h).view(B*N,D)

class VideoEncoder(nn.Module):
    def __init__(self,in_d=768,out_d=128,drop=.3):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_d,512),nn.LayerNorm(512),nn.ReLU(),nn.Dropout(drop),
            nn.Linear(512,256),nn.LayerNorm(256),nn.ReLU(),nn.Dropout(drop),
            nn.Linear(256,out_d),nn.ReLU())
    def forward(self,v):
        if v.dim()==3: v=v.squeeze(1)
        elif v.dim()==1: v=v.unsqueeze(0)
        return self.net(v)

class GatedFusion(nn.Module):
    def __init__(self,d=128,drop=.1):
        super().__init__(); self.sc=d**-.5
        self.eq=nn.Linear(d,d,bias=False); self.vk=nn.Linear(d,d,bias=False)
        self.vv=nn.Linear(d,d,bias=False); self.vq=nn.Linear(d,d,bias=False)
        self.ek=nn.Linear(d,d,bias=False); self.ev=nn.Linear(d,d,bias=False)
        self.ge=nn.Sequential(nn.Linear(d,d),nn.Sigmoid())
        self.gv=nn.Sequential(nn.Linear(d,d),nn.Sigmoid())
        self.proj=nn.Linear(d*2,d); self.norm=nn.LayerNorm(d)
        self.drop=nn.Dropout(drop)
    def forward(self,e,v):
        ae=torch.sigmoid((self.eq(e)*self.vk(v)).sum(-1,keepdim=True)*self.sc)
        ea=e+ae*self.vv(v)
        av=torch.sigmoid((self.vq(v)*self.ek(e)).sum(-1,keepdim=True)*self.sc)
        va=v+av*self.ev(e)
        f=self.proj(torch.cat([self.ge(ea)*ea,self.gv(va)*va],dim=-1))
        return self.norm(self.drop(f)+e)

class PyramidalGCN(nn.Module):
    def __init__(self,in_dim=21,hidden=128,heads=8,dropout=.35,
                 use_band_attn=True,use_hierarchy=True,use_functional=True,
                 n_domains=0,lambda_grl=.3,
                 fusion="none",video_dim=768,
                 edge_dropout=0.,feat_dropout=0.):
        super().__init__()
        self.use_band_attn=use_band_attn; self.use_hierarchy=use_hierarchy
        self.use_functional=use_functional; self.n_domains=n_domains
        self.fusion=fusion; self.edp=edge_dropout

        if use_band_attn: self.band_attn=BandAttn()
        self.proj=nn.Linear(in_dim,hidden); self.pnorm=nn.LayerNorm(hidden)
        self.fdrop=nn.Dropout(feat_dropout) if feat_dropout>0 else nn.Identity()
        self.gcn1=GCNBlock(hidden,hidden,dropout)
        self.gcn2=GCNBlock(hidden,hidden,dropout)

        if use_hierarchy:
            self.meso=MesoAttn(hidden,min(4,heads),dropout*.5)
            if use_functional:
                nh=max(1,heads//2); hd=max(1,hidden//nh)
                self.gat=GATv2Conv(hidden,hd,heads=nh,dropout=dropout,
                                   edge_dim=1,concat=True,add_self_loops=False)
                self.gnorm=nn.LayerNorm(hidden)
                self.gproj=nn.Linear(nh*hd,hidden,bias=False)

        if fusion in ("early","late"):
            self.venc=VideoEncoder(video_dim,hidden,dropout*.85)
        if fusion=="early":
            self.fuse=GatedFusion(hidden,dropout*.3)
            self.expand=nn.Sequential(nn.Linear(hidden,hidden*2),
                                      nn.LayerNorm(hidden*2),nn.ReLU(),
                                      nn.Dropout(dropout*.5))
        elif fusion=="late":
            self.fuse_mlp=nn.Sequential(nn.Linear(hidden*2,hidden*2),
                                        nn.LayerNorm(hidden*2),nn.ReLU(),
                                        nn.Dropout(dropout*.5))

        hi=hidden*2 if fusion in ("early","late") else hidden
        self.hval=nn.Linear(hi,2); self.haro=nn.Linear(hi,2)
        self.hdom=nn.Linear(hi,2); self.hlik=nn.Linear(hi,2)
        self.hqad=nn.Linear(hi,4)

        if n_domains>0:
            self.grl=GRL(lambda_grl)
            self.dom_cls=nn.Sequential(nn.Linear(hidden,64),nn.ReLU(),
                                       nn.Dropout(.3),nn.Linear(64,n_domains))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _edrop(self,ei,ew):
        if self.training and self.edp>0 and ei.shape[1]>0:
            mask=torch.rand(ei.shape[1],device=ei.device)>=self.edp
            return ei[:,mask],ew[mask]
        return ei,ew

    def forward_features(self,data):
        x=data.x; batch=data.batch
        if self.use_band_attn: x=self.band_attn(x)
        h=self.fdrop(F.relu(self.pnorm(self.proj(x))))
        ei_s,ew_s=self._edrop(data.edge_index_struct,data.edge_weight_struct)
        h=self.gcn1(h,ei_s,ew_s); h=self.gcn2(h,ei_s,ew_s)
        if self.use_hierarchy:
            h=self.meso(h,batch)
            if self.use_functional and data.edge_index_func.shape[1]>0:
                ei_f,ew_f=self._edrop(data.edge_index_func,data.edge_weight_func)
                hg=self.gat(h,ei_f,edge_attr=ew_f.unsqueeze(-1))
                h=self.gnorm(h+self.gproj(hg))
        return global_mean_pool(h,batch)

    def forward(self,data):
        g=self.forward_features(data)
        if self.fusion in ("early","late"):
            v=data.video
            if v.dim()==3: v=v.squeeze(1)
            v=self.venc(v)
            fused=self.expand(self.fuse(g,v)) if self.fusion=="early" \
                  else self.fuse_mlp(torch.cat([g,v],dim=-1))
        else:
            fused=g
        out={"val":self.hval(fused),"aro":self.haro(fused),
             "dom":self.hdom(fused),"lik":self.hlik(fused),
             "quad":self.hqad(fused)}
        if self.n_domains>0:
            out["domain"]=self.dom_cls(self.grl(g))
        return out

def build_model(mode="within",fusion="none",ablation="none"):
    abl=ABLATION_CONFIGS.get(ablation,ABLATION_CONFIGS["none"])
    loso=(mode=="loso")
    return PyramidalGCN(
        in_dim=abl["in_dim"], hidden=128,
        heads=8 if fusion=="none" else 4,
        dropout=.40 if loso else .35,
        use_band_attn=abl["use_band_attn"],
        use_hierarchy=abl["use_hierarchy"],
        use_functional=abl["use_functional"],
        n_domains=32 if (loso and fusion=="none") else 0,
        lambda_grl=.3, fusion=fusion,
        edge_dropout=.10 if loso else 0.,
        feat_dropout=.10 if loso else 0.,
    )

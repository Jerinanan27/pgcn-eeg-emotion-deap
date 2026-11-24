"""
identify_vit_from_h5.py
=======================
Tries to identify which pretrained ViT-B/16 produced the 768-d video embeddings
in fused_dataset.h5, using ONLY the stored embeddings (no videos needed).

Idea: if the embedding is the classifier input of a given ImageNet model (its
head replaced by an identity mapping), then applying THAT model's original
classification head to the embedding gives confident, sensible ImageNet
predictions (for face videos: e.g. "wig", "seat belt", "neck brace").
The head of a different model gives near-random, low-confidence predictions.
The head is linear, so this also works for the averaged frame embeddings.

Kaggle: Internet ON, input deap-fused-data. CPU is enough (about 5 minutes).
  pip install -q timm transformers
  python identify_vit_from_h5.py
"""
from pathlib import Path

import h5py
import numpy as np
import torch

H5 = next(Path("/kaggle/input").rglob("fused_dataset.h5"))
emb = []
with h5py.File(H5, "r") as f:
    for s in [f"s{i:02d}" for i in range(1, 23)]:
        for t in sorted(f[s].keys()):
            e = np.array(f[s][t]["video_embedding"], np.float32)
            if np.abs(e).sum() > 0:
                emb.append(e)
E = torch.from_numpy(np.stack(emb))
print(f"{len(E)} non-zero embeddings | mean norm {E.norm(dim=1).mean():.2f} | "
      f"mean {E.mean():.4f} | sd {E.std():.4f}")

heads = {}
import torchvision
for wname in ["IMAGENET1K_V1", "IMAGENET1K_SWAG_E2E_V1", "IMAGENET1K_SWAG_LINEAR_V1"]:
    try:
        w = getattr(torchvision.models.ViT_B_16_Weights, wname)
        m = torchvision.models.vit_b_16(weights=w)
        heads[f"torchvision {torchvision.__version__} vit_b_16 {wname}"] = (m.heads.head, w.meta["categories"])
    except Exception as ex:
        print("skip torchvision", wname, ex)
import timm
for name in ["vit_base_patch16_224.augreg2_in21k_ft_in1k", "vit_base_patch16_224.augreg_in21k_ft_in1k",
             "vit_base_patch16_224.augreg_in1k", "vit_base_patch16_224.orig_in21k_ft_in1k"]:
    try:
        m = timm.create_model(name, pretrained=True)
        heads[f"timm {timm.__version__} {name}"] = (m.get_classifier(), None)
    except Exception as ex:
        print("skip timm", name, ex)
try:
    import transformers
    from transformers import ViTForImageClassification
    m = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    heads[f"transformers {transformers.__version__} google/vit-base-patch16-224"] = (
        m.classifier, [m.config.id2label[i] for i in range(1000)])
except Exception as ex:
    print("skip transformers", ex)

cats = next(c for _, c in heads.values() if c is not None)   # ImageNet-1k class names
print("\nModel head applied to the stored embeddings:")
print(f"{'mean max-prob':>13} {'top class share':>15}  model  -> most frequent predicted classes")
rows = []
with torch.no_grad():
    for name, (head, c) in heads.items():
        p = torch.softmax(head(E), dim=1)
        top = p.argmax(1)
        vals, cnt = torch.unique(top, return_counts=True)
        order = cnt.argsort(descending=True)[:4]
        names = ", ".join(f"{(c or cats)[int(vals[i])]} ({int(cnt[i])})" for i in order)
        rows.append((p.max(1).values.mean().item(), cnt.max().item() / len(E), name, names))
for mp, share, name, names in sorted(rows, reverse=True):
    print(f"{mp:13.3f} {share:15.2f}  {name}  ->  {names}")
print("\nA clear winner has a much higher mean max-prob than the others and face-related "
      "classes. If all values are low and similar, the embeddings did not come from any of "
      "these classifiers (e.g. an ImageNet-21k or CLIP backbone), and the paper keeps its "
      "current wording.")

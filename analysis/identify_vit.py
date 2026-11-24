"""
identify_vit.py
===============
Finds out how the 768-d video embeddings in fused_dataset.h5 were made, without
asking the person who made them. It recomputes the embedding of a few trials
with each plausible recipe and compares it with the stored one (cosine
similarity). The recipe that reproduces the stored vector (similarity about
0.99 or higher) answers all three open questions at once: which ViT-B/16
weights, whether frames were face-cropped, and how many frames were used.

Needs on Kaggle: internet ON (to download model weights), a GPU is optional,
and these inputs: deap-fused-data (fused_dataset.h5) and the DEAP face videos
(face_video folder with s01/s01_trial01.avi ...). About 10 to 20 minutes.

  pip install timm transformers opencv-python-headless h5py
  python identify_vit.py
"""
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

IN = Path("/kaggle/input")
H5 = next(IN.rglob("fused_dataset.h5"))
VIDEOS = {p.stem.lower(): p for p in IN.rglob("*.avi")}
TRIALS = [("s01", 1), ("s05", 12), ("s14", 30)]          # three test trials
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("H5:", H5, "| videos found:", len(VIDEOS))

face_det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def frames_1fps(path, max_frames=60):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 50
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps)))
    out = []
    for i in range(0, n, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        if len(out) == max_frames:
            break
    cap.release()
    return out, fps, n


def crop_face(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_det.detectMultiScale(g, 1.1, 5)
    if len(faces) == 0:
        h, w = img.shape[:2]; s = min(h, w)
        return img[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return img[y:y + h, x:x + w]


def to_tensor(imgs, mean, std):
    arr = np.stack([cv2.resize(i, (224, 224), interpolation=cv2.INTER_LINEAR) for i in imgs]).astype(np.float32) / 255.0
    arr = (arr - np.array(mean, np.float32)) / np.array(std, np.float32)
    return torch.from_numpy(arr).permute(0, 3, 1, 2)


def load_models():
    models = {}
    import torchvision
    m = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
    m.heads = torch.nn.Identity()
    models["torchvision vit_b_16 IMAGENET1K_V1"] = (m, lambda m, x: m(x), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    import timm
    for name in ["vit_base_patch16_224.augreg2_in21k_ft_in1k", "vit_base_patch16_224.augreg_in21k"]:
        try:
            t = timm.create_model(name, pretrained=True, num_classes=0)
            cfg = t.pretrained_cfg
            models[f"timm {name}"] = (t, lambda m, x: m(x), cfg["mean"], cfg["std"])
        except Exception as e:
            print("skip", name, e)
    from transformers import ViTModel
    for name in ["google/vit-base-patch16-224", "google/vit-base-patch16-224-in21k"]:
        try:
            h = ViTModel.from_pretrained(name, add_pooling_layer=True)
            models[f"HF {name} CLS"] = (h, lambda m, x: m(pixel_values=x).last_hidden_state[:, 0], (0.5,) * 3, (0.5,) * 3)
            models[f"HF {name} pooler"] = (h, lambda m, x: m(pixel_values=x).pooler_output, (0.5,) * 3, (0.5,) * 3)
        except Exception as e:
            print("skip", name, e)
    try:
        import open_clip  # optional
    except ImportError:
        pass
    return models


@torch.no_grad()
def main():
    models = load_models()
    with h5py.File(H5, "r") as f:
        stored = {(s, t): np.array(f[s][f"trial_{t:02d}"]["video_embedding"], np.float32) for s, t in TRIALS}
    results = {}
    for s, t in TRIALS:
        key = f"{s}_trial{t:02d}"
        if key not in VIDEOS:
            print("video not found:", key); continue
        frames, fps, n = frames_1fps(VIDEOS[key], max_frames=10_000)
        print(f"{key}: {n} frames at {fps:.1f} fps = {n / fps:.1f} s -> {len(frames)} frames at 1 fps")
        for crop in (False, True):
            imgs = [crop_face(i) for i in frames] if crop else frames
            for nfr in (60, len(imgs)):
                for mname, (m, fwd, mean, std) in models.items():
                    m.eval().to(DEV)
                    x = to_tensor(imgs[:nfr], mean, std).to(DEV)
                    e = torch.cat([fwd(m, x[i:i + 16]) for i in range(0, len(x), 16)]).mean(0).cpu().numpy()
                    v = stored[(s, t)]
                    cos = float(e @ v / (np.linalg.norm(e) * np.linalg.norm(v) + 1e-12))
                    results.setdefault((mname, crop, nfr == 60), []).append(cos)
    print("\nMean cosine similarity with the stored embeddings (1.00 = same recipe):")
    for (mname, crop, first60), c in sorted(results.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"{np.mean(c):.4f}  {mname:45s} face-crop={crop!s:5s} frames={'first 60' if first60 else 'all'}")


if __name__ == "__main__":
    main()

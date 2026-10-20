"""Quick live demo: load a trained checkpoint, predict on held-out test images, show metrics.

python demo.py --dataset kvasir   # polyp segmentation (results/main_bce_dice/best.pth)
python demo.py --dataset u373     # PhC-C2DH-U373 cells, dataset from the U-Net paper (results/u373/best.pth)
Options: --n 4 (images shown), --seed 0 (which images), --no-show (only save results/demo_<dataset>.png)
"""
import argparse, sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "Pytorch-UNet"))
sys.path.insert(0, str(ROOT))
from unet import UNet  # noqa: E402
from run_unet import evaluate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["kvasir", "u373"], default="kvasir")
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-show", action="store_true")
a = ap.parse_args()
if a.no_show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if a.dataset == "kvasir":
    from run_unet import DS, TEST_PQ, load
    X, Y = load(TEST_PQ)
    ds, n_ch, ckpt = DS(X, Y), 3, ROOT / "results/main_bce_dice/best.pth"
else:
    from run_u373 import DS, load_seq
    X2, Y2 = load_seq("02")
    te = np.random.RandomState(0).permutation(len(X2))[4:]  # same test split as training script
    ds, n_ch, ckpt = DS(X2[te], Y2[te]), 1, ROOT / "results/u373/best.pth"

model = UNet(n_channels=n_ch, n_classes=1, bilinear=False).to(dev)
model.load_state_dict(torch.load(ckpt, map_location=dev))
model.eval()

print(f"Dataset: {a.dataset} | test images: {len(ds)} | device: {dev}")
m = evaluate(model, DataLoader(ds, 8), dev)
print("Test-set metrics:")
for k in ["dice", "iou", "pixel_accuracy", "precision", "recall"]:
    print(f"  {k:15s} {m[k]:.4f}")

idx = np.random.RandomState(a.seed).choice(len(ds), min(a.n, len(ds)), replace=False)
fig, ax = plt.subplots(len(idx), 4, figsize=(11, 2.8 * len(idx)))
ax = np.atleast_2d(ax)
for r, i in enumerate(idx):
    x, y = ds[i]
    with torch.no_grad():
        p = (torch.sigmoid(model(x[None].to(dev)))[0, 0] > 0.5).cpu().numpy()
    gt = y[0].numpy()
    img = x.permute(1, 2, 0).numpy() if n_ch == 3 else x[0].numpy()
    if n_ch == 1:
        img = (img - img.min()) / (img.max() - img.min())
    base = img if n_ch == 3 else np.stack([img] * 3, -1)
    ov = base.copy(); ov[p > 0] = 0.5 * ov[p > 0] + 0.5 * np.array([1, 0, 0])
    inter, union = (p * gt).sum(), p.sum() + gt.sum()
    dice = (2 * inter + 1e-6) / (union + 1e-6)
    for c, (im, t) in enumerate([(img, "Input"), (gt, "True mask"), (p, f"Predicted (Dice {dice:.2f})"), (ov, "Overlay")]):
        ax[r, c].imshow(im, cmap="gray"); ax[r, c].axis("off"); ax[r, c].set_title(t, fontsize=9)
fig.suptitle(f"U-Net demo - {a.dataset}", y=0.995)
fig.tight_layout()
out = ROOT / "results" / f"demo_{a.dataset}.png"
fig.savefig(out, dpi=110)
print("Saved", out)
if not a.no_show:
    plt.show()

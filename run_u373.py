"""U-Net on PhC-C2DH-U373 (Cell Tracking Challenge), one of the datasets used in the original U-Net paper.

Train on gold-standard frames of sequence 01, validate/test on sequence 02 (unseen sequence).
Usage: python run_u373.py --epochs 150 --name u373
Outputs go to results/<name>/ like run_unet.py.
"""
import argparse, glob, json, random, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "Pytorch-UNet"))
from unet import UNet  # noqa: E402
from run_unet import dice_loss, evaluate  # noqa: E402

BASE = ROOT / "data/u373/raw/PhC-C2DH-U373"
H, W = 384, 512


def load_seq(seq):
    xs, ys = [], []
    for g in sorted(glob.glob(str(BASE / f"{seq}_GT/SEG/man_seg*.tif"))):
        idx = Path(g).stem[-3:]
        img = tifffile.imread(BASE / seq / f"t{idx}.tif").astype(np.float32)
        msk = (tifffile.imread(g) > 0).astype(np.uint8) * 255
        img = np.asarray(Image.fromarray(img).resize((W, H), Image.BILINEAR))
        img = (img - img.mean()) / (img.std() + 1e-6)
        msk = np.asarray(Image.fromarray(msk).resize((W, H), Image.NEAREST)) > 127
        xs.append(img); ys.append(msk.astype(np.float32))
    return np.stack(xs), np.stack(ys)


class DS(Dataset):
    def __init__(self, x, y, aug=False):
        self.x, self.y, self.aug = x, y, aug

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = torch.from_numpy(self.x[i])[None]
        y = torch.from_numpy(self.y[i])[None]
        if self.aug:
            if random.random() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if random.random() < 0.5:
                x, y = x.flip(-2), y.flip(-2)
            x = x * random.uniform(0.8, 1.2) + random.uniform(-0.2, 0.2)
        return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--name", default="u373")
    a = ap.parse_args()

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ROOT / "results" / a.name
    out.mkdir(parents=True, exist_ok=True)

    X1, Y1 = load_seq("01")
    X2, Y2 = load_seq("02")
    perm = np.random.RandomState(0).permutation(len(X2))
    va, te = perm[:4], perm[4:]
    train_dl = DataLoader(DS(X1, Y1, aug=True), a.bs, shuffle=True)
    val_dl = DataLoader(DS(X2[va], Y2[va]), a.bs)
    test_ds = DS(X2[te], Y2[te])
    test_dl = DataLoader(test_ds, a.bs)
    print(f"train {len(X1)} (seq 01)  val {len(va)}  test {len(te)} (seq 02)  device {dev}", flush=True)

    model = UNet(n_channels=1, n_classes=1, bilinear=False).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    hist = {"train_loss": [], "val_dice": []}
    best = -1.0

    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item() * len(x)
        vd = evaluate(model, val_dl, dev)["dice"]
        hist["train_loss"].append(tot / len(X1)); hist["val_dice"].append(vd)
        if vd > best:
            best = vd
            torch.save(model.state_dict(), out / "best.pth")
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep:3d}  loss {hist['train_loss'][-1]:.4f}  val dice {vd:.4f}", flush=True)

    model.load_state_dict(torch.load(out / "best.pth"))
    metrics = evaluate(model, test_dl, dev)
    metrics.update(config=vars(a), best_val_dice=best)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("TEST:", json.dumps({k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(hist["train_loss"]); ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(hist["val_dice"]); ax[1].set_title("Validation Dice"); ax[1].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(out / "curves.png", dpi=150); plt.close(fig)

    idx = np.random.RandomState(1).choice(len(test_ds), 5, replace=False)
    fig, ax = plt.subplots(len(idx), 4, figsize=(12, 2.9 * len(idx)))
    model.eval()
    for r, i in enumerate(idx):
        x, y = test_ds[i]
        with torch.no_grad():
            p = (torch.sigmoid(model(x[None].to(dev)))[0, 0] > 0.5).cpu().numpy()
        img = x[0].numpy(); img = (img - img.min()) / (img.max() - img.min())
        gt = y[0].numpy()
        ov = np.stack([img] * 3, -1); ov[p > 0] = 0.5 * ov[p > 0] + 0.5 * np.array([1, 0, 0])
        for c, (im, t) in enumerate([(img, "Input"), (gt, "True mask"), (p, "Predicted mask"), (ov, "Overlay")]):
            ax[r, c].imshow(im, cmap="gray"); ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(t)
    fig.tight_layout(); fig.savefig(out / "predictions.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()

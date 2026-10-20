"""Train/evaluate milesial's Pytorch-UNet on Kvasir-SEG.

Usage: python run_unet.py --loss bce|bce_dice --epochs 30 --name exp1 [--no-aug]
Outputs go to results/<name>/ (metrics.json, curves.png, predictions.png, best.pth).
"""
import argparse, io, json, random, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "Pytorch-UNet"))
from unet import UNet  # noqa: E402  (original repo code, unmodified)

SIZE = 256
TRAIN_PQ = ROOT / "data/train-00000-of-00001-7a33e3aba9046bad.parquet"
TEST_PQ = ROOT / "data/validation-00000-of-00001-4d65c2d2950bdc96.parquet"


def load(pq):
    df = pd.read_parquet(pq)
    imgs, masks = [], []
    for _, r in df.iterrows():
        im = Image.open(io.BytesIO(r.image["bytes"])).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        m = Image.open(io.BytesIO(r.annotation["bytes"])).convert("L").resize((SIZE, SIZE), Image.NEAREST)
        imgs.append(np.asarray(im, dtype=np.uint8))
        masks.append((np.asarray(m) > 127).astype(np.float32))
    return np.stack(imgs), np.stack(masks)


class DS(Dataset):
    def __init__(self, x, y, aug=False):
        self.x, self.y, self.aug = x, y, aug

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = torch.from_numpy(self.x[i]).permute(2, 0, 1).float() / 255
        y = torch.from_numpy(self.y[i])[None]
        if self.aug:
            if random.random() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if random.random() < 0.5:
                x, y = x.flip(-2), y.flip(-2)
            k = random.randint(0, 3)
            x, y = torch.rot90(x, k, (1, 2)), torch.rot90(y, k, (1, 2))
            x = (x * random.uniform(0.8, 1.2)).clamp(0, 1)
        return x, y


def dice_loss(logits, y, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + eps)).mean()


@torch.no_grad()
def evaluate(model, loader, dev):
    tp = fp = fn = tn = 0.0
    dices, ious = [], []
    model.eval()
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        pred = (torch.sigmoid(model(x)) > 0.5).float()
        tp += (pred * y).sum().item(); fp += (pred * (1 - y)).sum().item()
        fn += ((1 - pred) * y).sum().item(); tn += ((1 - pred) * (1 - y)).sum().item()
        inter = (pred * y).sum((1, 2, 3)); union = pred.sum((1, 2, 3)) + y.sum((1, 2, 3))
        dices.append(((2 * inter + 1e-6) / (union + 1e-6)))
        ious.append(((inter + 1e-6) / (union - inter + 1e-6)))
    d, j = torch.cat(dices).mean().item(), torch.cat(ious).mean().item()
    return {
        "dice": d, "iou": j,
        "pixel_accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": tp / (tp + fp + 1e-9), "recall": tp / (tp + fn + 1e-9),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["bce", "bce_dice"], default="bce_dice")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--name", default="exp1")
    a = ap.parse_args()

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ROOT / "results" / a.name
    out.mkdir(parents=True, exist_ok=True)

    X, Y = load(TRAIN_PQ)
    Xt, Yt = load(TEST_PQ)
    perm = np.random.RandomState(0).permutation(len(X))
    tr, va = perm[:800], perm[800:]
    train_dl = DataLoader(DS(X[tr], Y[tr], aug=not a.no_aug), a.bs, shuffle=True)
    val_dl = DataLoader(DS(X[va], Y[va]), a.bs)
    test_ds = DS(Xt, Yt)
    test_dl = DataLoader(test_ds, a.bs)
    print(f"train {len(tr)}  val {len(va)}  test {len(Xt)}  device {dev}", flush=True)

    model = UNet(n_channels=3, n_classes=1, bilinear=False).to(dev)
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
                loss = F.binary_cross_entropy_with_logits(logits, y)
                if a.loss == "bce_dice":
                    loss = loss + dice_loss(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item() * len(x)
        vd = evaluate(model, val_dl, dev)["dice"]
        hist["train_loss"].append(tot / len(tr)); hist["val_dice"].append(vd)
        if vd > best:
            best = vd
            torch.save(model.state_dict(), out / "best.pth")
        print(f"epoch {ep:2d}  loss {hist['train_loss'][-1]:.4f}  val dice {vd:.4f}", flush=True)

    model.load_state_dict(torch.load(out / "best.pth"))
    metrics = evaluate(model, test_dl, dev)
    metrics.update(config=vars(a), best_val_dice=best)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("TEST:", json.dumps({k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(hist["train_loss"]); ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(hist["val_dice"]); ax[1].set_title("Validation Dice"); ax[1].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(out / "curves.png", dpi=150); plt.close(fig)

    # image panels: input | ground truth | prediction | overlay
    idx = np.random.RandomState(1).choice(len(Xt), 6, replace=False)
    fig, ax = plt.subplots(len(idx), 4, figsize=(10, 2.6 * len(idx)))
    model.eval()
    for r, i in enumerate(idx):
        x, y = test_ds[i]
        with torch.no_grad():
            p = (torch.sigmoid(model(x[None].to(dev)))[0, 0] > 0.5).cpu().numpy()
        img = x.permute(1, 2, 0).numpy()
        gt = y[0].numpy()
        ov = img.copy(); ov[p > 0] = 0.5 * ov[p > 0] + 0.5 * np.array([1, 0, 0])
        for c, (im, t) in enumerate([(img, "Input"), (gt, "True mask"), (p, "Predicted mask"), (ov, "Overlay")]):
            ax[r, c].imshow(im, cmap="gray"); ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(t)
    fig.tight_layout(); fig.savefig(out / "predictions.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()

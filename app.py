"""Segmentation Lab: Streamlit demo for the U-Net replication.

Run: streamlit run app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "Pytorch-UNet"))
sys.path.insert(0, str(ROOT))
from unet import UNet  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset -> (input channels, {model label: results folder})
CONFIG = {
    "Cells": (1, {"BCE + Dice": "u373"}),
    "Polyps": (3, {"BCE + Dice": "main_bce_dice", "BCE only": "extra_bce_only"}),
}
DESCRIPTION = {
    "Cells": "PhC-C2DH-U373 (dataset from the U-Net paper). Tested on an unseen sequence.",
    "Polyps": "Kvasir-SEG endoscopy polyps (extra dataset). Compares two training losses.",
}

st.set_page_config(page_title="Segmentation Lab", layout="wide")
st.markdown(
    """<style>
    .block-container {padding-top: 2rem; max-width: 1100px;}
    div[data-testid="stMetric"] {background:#000; border-radius:10px; padding:12px 16px;}
    div[data-testid="stImage"] img {border-radius:8px;}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading test data...")
def load_test(dataset):
    if dataset == "Cells":
        from run_u373 import load_seq
        X, Y = load_seq("02")
        te = np.random.RandomState(0).permutation(len(X))[4:]  # same split as training
        return X[te], Y[te]
    from run_unet import TEST_PQ, load
    return load(TEST_PQ)


@st.cache_resource(show_spinner="Loading model...")
def load_model(dataset, folder):
    model = UNet(n_channels=CONFIG[dataset][0], n_classes=1, bilinear=False).to(DEV)
    model.load_state_dict(torch.load(ROOT / "results" / folder / "best.pth", map_location=DEV))
    return model.eval()


def to_tensor(dataset, arr):
    """Stored test array -> model input tensor (1, C, H, W)."""
    if dataset == "Cells":
        return torch.from_numpy(arr)[None, None]
    return torch.from_numpy(arr).permute(2, 0, 1)[None].float() / 255


def predict(model, x):
    with torch.no_grad():
        return (torch.sigmoid(model(x.to(DEV)))[0, 0] > 0.5).cpu().numpy()


def display_image(dataset, arr):
    if dataset == "Cells":
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
    return arr / 255.0


def overlay(img, pred):
    rgb = np.stack([img] * 3, -1) if img.ndim == 2 else img.copy()
    rgb[pred > 0] = 0.5 * rgb[pred > 0] + 0.5 * np.array([0.9, 0.1, 0.1])
    return rgb


def sample_metrics(pred, gt):
    tp = float((pred * gt).sum()); fp = float((pred * (1 - gt)).sum())
    fn = float(((1 - pred) * gt).sum()); tn = float(((1 - pred) * (1 - gt)).sum())
    return {
        "Dice": (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6),
        "IoU": (tp + 1e-6) / (tp + fp + fn + 1e-6),
        "Precision": tp / (tp + fp + 1e-9),
        "Recall": tp / (tp + fn + 1e-9),
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
    }


def panel(title, image, caption, **kw):
    with st.container(border=True):
        st.subheader(title)
        st.image(image, caption=caption, width="stretch", **kw)


st.title("Segmentation Lab")
st.caption("Research demo · U-Net (Ronneberger et al., MICCAI 2015) replication · image segmentation and evaluation")
st.divider()

c1, c2, c3 = st.columns(3)
dataset = c1.selectbox("Dataset", list(CONFIG))
model_label = c2.selectbox("Model", list(CONFIG[dataset][1]))
folder = CONFIG[dataset][1][model_label]
X, Y = load_test(dataset)
sample = c3.selectbox("Sample", [f"Test #{i + 1}" for i in range(len(X))] + ["Upload my own image"])
st.caption(DESCRIPTION[dataset])

model = load_model(dataset, folder)
gt = None
if sample == "Upload my own image":
    up = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if up is None:
        st.info("Upload an image to segment it. No ground truth is available for uploads.")
        st.stop()
    im = Image.open(up)
    if dataset == "Cells":
        a = np.asarray(im.convert("L").resize((512, 384), Image.BILINEAR), dtype=np.float32)
        arr = (a - a.mean()) / (a.std() + 1e-6)
    else:
        arr = np.asarray(im.convert("RGB").resize((256, 256), Image.BILINEAR), dtype=np.uint8)
else:
    i = int(sample.split("#")[1]) - 1
    arr, gt = X[i], Y[i]

pred = predict(model, to_tensor(dataset, arr))
img = display_image(dataset, arr)

a, b = st.columns(2)
with a:
    panel("Input image", img, "Original sample", clamp=True)
with b:
    if gt is not None:
        panel("Ground-truth mask", gt, "Actual segmentation", clamp=True)
    else:
        panel("Ground-truth mask", np.zeros_like(pred, dtype=float), "Not available for uploads", clamp=True)
a, b = st.columns(2)
with a:
    panel("Predicted mask", pred.astype(float), "Model output", clamp=True)
with b:
    panel("Overlay", overlay(img, pred), "Red mask on original", clamp=True)

st.subheader("Evaluation metrics")
m = json.loads((ROOT / "results" / folder / "metrics.json").read_text())
st.caption(f"Full test set ({len(X)} images) · model: {model_label}")
t = st.columns(3)
t[0].metric("Dice", f"{m['dice']:.3f}")
t[1].metric("IoU", f"{m['iou']:.3f}")
t[2].metric("Precision", f"{m['precision']:.3f}")
t = st.columns(3)
t[0].metric("Recall", f"{m['recall']:.3f}")
t[1].metric("Accuracy", f"{m['pixel_accuracy']:.3f}")
if gt is not None:
    t[2].metric("Image Dice (this sample)", f"{sample_metrics(pred, gt)['Dice']:.3f}")
else:
    t[2].metric("Image Dice (this sample)", "n/a")

if dataset == "Polyps":
    with st.expander("Extra experiment: BCE vs BCE + Dice loss"):
        rows = {}
        for lbl, f in CONFIG["Polyps"][1].items():
            r = json.loads((ROOT / "results" / f / "metrics.json").read_text())
            rows[lbl] = {k: round(r[k], 3) for k in ["dice", "iou", "pixel_accuracy", "precision", "recall"]}
        st.table(rows)

"""Turn an arbitrary uploaded image (and optional mask) into the array format the models were trained on."""
import numpy as np
from PIL import Image


def prepare_image(dataset, im):
    """Cells: grayscale 384x512, standardised float32. Polyps: RGB 256x256 uint8."""
    if dataset == "Cells":
        a = np.asarray(im.convert("L").resize((512, 384), Image.BILINEAR), dtype=np.float32)
        return (a - a.mean()) / (a.std() + 1e-6)
    return np.asarray(im.convert("RGB").resize((256, 256), Image.BILINEAR), dtype=np.uint8)


def prepare_mask(dataset, im):
    size = (512, 384) if dataset == "Cells" else (256, 256)
    m = np.asarray(im.convert("L").resize(size, Image.NEAREST))
    return (m > 127).astype(np.float32)

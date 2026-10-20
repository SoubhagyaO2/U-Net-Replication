# U-Net Replication: Polyp Segmentation on Kvasir-SEG

Replication of **U-Net: Convolutional Networks for Biomedical Image Segmentation**
(Ronneberger, Fischer, Brox, MICCAI 2015, Springer LNCS).
DOI: https://doi.org/10.1007/978-3-319-24574-4_28

The U-Net implementation in `Pytorch-UNet/` is [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
(MIT license), used unmodified. `run_unet.py` is our own training/evaluation script.

## Experiment 1: PhC-C2DH-U373 (dataset used in the original paper)
Cell Tracking Challenge data (http://celltrackingchallenge.net). Gold-standard annotated frames: 15 from sequence 01
(train) and 19 from sequence 02 (4 validation, 15 test), so the test set comes from an unseen sequence.
Semantic (cell vs. background) segmentation, 384x512 input, BCE + Dice loss, 150 epochs.
Note: the paper does instance segmentation with a weighted loss to split touching cells; this replication does not.

| Dice | IoU | Pixel acc. | Precision | Recall |
|---|---|---|---|---|
| 0.891 | 0.815 | 0.989 | 0.860 | 0.943 |

```
python run_u373.py --epochs 150 --name u373
python demo.py --dataset u373      # fast demo from saved checkpoint
```
Download the data from the Cell Tracking Challenge (PhC-C2DH-U373 training set) and unzip into `data/u373/raw/`.

## Experiment 2: Kvasir-SEG polyp segmentation (extra dataset)
- Dataset: Kvasir-SEG (1000 polyp images + masks), resized to 256x256. 800 train / 80 val / 120 held-out test.
  Not included in this repo; obtain it from https://datasets.simula.no/kvasir-seg/ or the Hugging Face
  mirror `kowndinya23/Kvasir-SEG` and place the parquet files in `data/` (paths are set at the top of `run_unet.py`).
- Model: UNet(3 -> 1 channel), transposed-conv upsampling, AdamW lr 1e-4, batch 8, 40 epochs, mixed precision,
  flip/rotate/brightness augmentation. Trained on an RTX 3050 Ti (4 GB).

```
pip install torch torchvision pandas pyarrow pillow numpy matplotlib
python run_unet.py --loss bce_dice --epochs 40 --name main_bce_dice
python run_unet.py --loss bce      --epochs 40 --name extra_bce_only
```

## Results (120 held-out test images)

| Metric | BCE + Dice | BCE only |
|---|---|---|
| Dice | **0.791** | 0.757 |
| IoU | **0.698** | 0.671 |
| Pixel accuracy | **0.941** | 0.940 |
| Precision | 0.844 | **0.867** |
| Recall | **0.751** | 0.718 |

Extra experiment: adding a Dice term to the BCE loss improves Dice by ~3.4 points and IoU by ~2.7.

Outputs per run are in `results/<name>/`: `predictions.png` (input / true mask / predicted mask / overlay),
`curves.png`, `metrics.json`. Model checkpoints (`best.pth`) are not committed.

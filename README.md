# ViT-GRU for HMDB51 Action Recognition

PyTorch implementation of a **Vision Transformer + GRU** hybrid model for video action recognition on the [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) dataset.

This repository is a clean, modular conversion of an original Kaggle notebook.

## Architecture

Three model variants are provided for ablation studies:

| Model | Description |
|-------|-------------|
| `vit_gru` | ViT (frame features) → GRU (temporal modeling) → classifier |
| `vit_only` | ViT + temporal average pooling (no RNN) |
| `gru_fe` | Generic feature extractor + GRU |

Default backbone: `vit_small_patch16_224` (from [timm](https://github.com/huggingface/pytorch-image-models)).

## Project Structure

```
vit-gru-hmdb51/
├── src/
│   ├── __init__.py
│   ├── dataset.py      # HMDB51Dataset + transforms
│   ├── models.py       # ViT_GRU, ViT_Only, FocalLoss, etc.
│   ├── train.py        # train / evaluate / test loops
│   └── utils.py        # metrics, scheduler, plotting, data prep
├── scripts/
│   ├── prepare_data.py # Create train/val/test CSV splits
│   ├── train.py        # Training entry point
│   └── evaluate.py     # Evaluation + confusion matrix
├── data/               # Place your CSV / (optional) videos here
├── outputs/            # Checkpoints, plots, metrics
├── notebooks/          # Original notebook (optional)
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/vit-gru-hmdb51.git
cd vit-gru-hmdb51

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **Note:** A CUDA-capable GPU is strongly recommended.

## Dataset

1. Download [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).
2. Expected layout:
   ```
   hmdb51/
   ├── brush_hair/
   │   ├── video1.avi
   │   └── ...
   ├── cartwheel/
   └── ... (51 classes)
   ```
3. Create splits CSV:

```bash
python scripts/prepare_data.py \
    --video_dir /path/to/hmdb51 \
    --output_csv data/hmdb51_splits.csv
```

Default split ratios: **70% train / 15% val / 15% test** (per class, shuffled).

## Training

```bash
python scripts/train.py \
    --csv data/hmdb51_splits.csv \
    --model vit_gru \
    --epochs 10 \
    --batch_size 4 \
    --lr 1e-4 \
    --frames 32 \
    --output_dir outputs/vit_gru
```

Useful flags:

- `--model {vit_gru,vit_only,gru_fe}`
- `--resume path/to/last_checkpoint.pth`
- `--warmup_epochs 5`

Checkpoints are saved to `--output_dir`:
- `best_model.pth` – highest validation accuracy
- `last_checkpoint.pth` – most recent epoch

## Evaluation

```bash
python scripts/evaluate.py \
    --csv data/hmdb51_splits.csv \
    --checkpoint outputs/vit_gru/best_model.pth \
    --output_dir outputs/vit_gru
```

Produces:
- `confusion_matrix.png`
- `class_wise_accuracy.png`
- `test_metrics.txt`

## Key Implementation Details

- **Frame sampling**: Uniformly sample 32 frames per video (pad by repeating if shorter).
- **Augmentation** (train only): random horizontal/vertical flip, rotation, color jitter, random resized crop.
- **Loss**: Focal Loss (`γ=2`) to mitigate class imbalance.
- **Optimizer**: Adam + weight decay + linear warmup + cosine annealing.
- **ViT processing**: `timm` backbone; optional `transformers.ViTImageProcessor` for compatibility.

## Citation / Acknowledgement

If you use this code, please consider citing the original papers:

- Dosovitskiy et al., *An Image is Worth 16x16 Words*, ICLR 2021 (ViT)
- Kuehne et al., *HMDB: A Large Video Database for Human Motion Recognition*, ICCV 2011

## License

MIT License – feel free to use and modify.

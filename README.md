# GOAL: Global Self-Sustaining and Local Inheritance

[中文说明](README.zh-CN.md) · [Paper](https://doi.org/10.1016/j.patcog.2024.110679) · [MIT License](LICENSE)

PyTorch training code for **Global self-sustaining and local inheritance for source-free unsupervised domain adaptation**, Pattern Recognition 155 (2024), 110679.

**Authors:** Lin Peng, Yuhang He, Shaokun Wang, Xiang Song, Songlin Dong, Xing Wei, and Yihong Gong.

GOAL adapts a source-trained classifier using only unlabeled target-domain images. It combines a mean teacher, Global Consistency Self-Sustaining (GCSS), and Local Topology Inheritance (LTI).

This release provides an **Office-Home / ImageNet V2** preset. Dataset images, trained checkpoints, and completed experiment runs are not included. The paper reports 74.1% average accuracy on Office-Home; that number is a paper reference, not a verified result of this release.

## Requirements

- Linux, or a suitably configured WSL2 environment with NVIDIA GPU support. The batch scheduler uses Unix process and file-locking interfaces; native Windows is not supported by that scheduler.
- Python 3.10 and an NVIDIA driver compatible with the pinned CUDA 11.7 PyTorch packages.
- PyTorch 2.0.0, torchvision 0.15.1, NumPy 1.26.3, and Pillow 11.3.0 (see [requirements.txt](requirements.txt)).

Run all commands below from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Prepare Office-Home

Obtain the dataset separately and arrange its class directories as follows:

```text
/path/to/officeHome/
├── Art/<class>/*.jpg
├── Clipart/<class>/*.jpg
├── Product/<class>/*.jpg
└── RealWorld/<class>/*.jpg
```

`Real_World/` is also accepted. Generate lists for your own dataset layout:

```bash
export DATA_ROOT=/path/to/officeHome
python scripts/prepare_data.py \
  --data-root "$DATA_ROOT" --output data/officehome --strict-counts
```

Expected counts are Art: 2,427; Clipart: 4,365; Product: 4,439; Real-world: 4,357, with 65 shared classes. Lists contain relative image paths and zero-based labels. Target labels are used only for evaluation, not adaptation losses or checkpoint selection. When importing an external source checkpoint, verify its class-to-label mapping; `prepare_data.py --class-map` accepts an explicit mapping.

## Train source models

Train all four source domains on physical GPU 0:

```bash
PYTHON="$(command -v python)" bash scripts/train_source.sh "$DATA_ROOT" 0
```

The script uses ResNet-50 `IMAGENET1K_V2`, all source images, 20 epochs, batch size 64, label smoothing 0.1, seed 2021, and final-epoch checkpoint selection. ImageNet weights are downloaded on the first run. Source models are written to `runs/source/{A,C,P,R}/`.

## Adapt to target domains

Run all 12 directed Office-Home transfers on one GPU:

```bash
python scripts/run_officehome.py \
  --data-root "$DATA_ROOT" \
  --source-root runs/source \
  --output-root runs/target \
  --gpus 0
```

Use `--gpus 0,1,2` to distribute independent tasks across physical GPUs. Use `--pairs A2C,C2A` for selected transfers, `--dry-run` to inspect planned commands after preparing source checkpoints, and `--resume` to recover unfinished runs from epoch-boundary checkpoints. Completed matching runs are skipped.

The [default configuration](configs/officehome_goal_v2.json) uses:

| Setting | Value |
|---|---|
| Target epochs / batch size | 40 / 64 |
| Bottleneck and classifier learning rate | 0.001 |
| Backbone learning-rate multiplier | 0.1 |
| Nearest neighbors / top feature dimensions | 5 / 5 |
| Teacher parameter EMA / prediction EMA | 0.9 / 0.9 |
| Separation coefficient / topology coefficient | 1.0 / 1.0 |
| Separation decay exponent | 0 (constant) |
| Random seed | 2021 |

The student is updated by gradient descent; teacher parameters and floating-point buffers are updated by EMA. The memory bank stores target features and smoothed predictions. Adaptation uses the final checkpoint rather than selecting a checkpoint by target accuracy.

## Evaluate and summarize

Each target run is stored under `runs/target/goal_v2/seed2021/<task>/`:

- `last.pt`: student, teacher, optimizer, memory-bank, and random states.
- `run.json`: configuration and input provenance.
- `metrics.jsonl` and `stdout.log`: training metrics and output.
- `summary.json`: completion status and final metrics.

```bash
python scripts/summarize.py --runs runs/target

python scripts/evaluate.py \
  --run runs/target/goal_v2/seed2021/A2C \
  --model student --device cuda:0
```

Student and teacher results are reported separately. A full benchmark average is reported only when every transfer task is available. Optional inference-time augmentation is exposed by `evaluate.py --tta flip` and `--tta ten_crop` for student checkpoints; these are separate evaluation protocols.

## Implementation choices

The current preset makes the following choices explicit. Account for them when comparing results with the paper or another implementation:

| Component | Current preset |
|---|---|
| Source initialization | torchvision `IMAGENET1K_V2` |
| Neighbor confidence | Shannon entropy of the memory-bank prediction; Eq. (4) is written as a cross-entropy expression |
| Separation reduction | `sum`: average over anchors, without Eq. (5)'s additional division by batch size minus one |
| Topology features | L2-normalized before computing distances |
| Topology mask / reduction | Valid confident same-class edges only; mean over valid undirected edges |
| Node reliability | Detached from the gradient computation |
| Entropy threshold | Once per epoch, from the latest stored instantaneous teacher prediction for each sample |

Alternative entropy and topology modes are implemented in `goal/losses.py`. Reciprocal-neighbor reweighting and an additional neighbor cross-entropy loss are optional extensions and are disabled in the shipped preset. Utilities also contain Office-31 and ImageCLEF support, but this release only includes an Office-Home configuration and image lists.

## Code map

| File | Purpose |
|---|---|
| `goal/models.py` | ResNet-50, 256-dimensional bottleneck, classifier, checkpoint conversion |
| `goal/source.py` | Supervised source training |
| `goal/train.py` | Target adaptation and checkpoint recovery |
| `goal/losses.py` | Neighbor reliability, attraction, separation, and topology losses |
| `goal/memory.py` | Feature/prediction banks and mean-teacher updates |
| `goal/data.py` | Image lists, augmentations, dataset preparation, and optional cache |
| `scripts/run_officehome.py` | Multi-GPU experiment scheduling |
| `scripts/build_image_cache.py` | Optional cache of resized input images |

## Citation

```bibtex
@article{peng2024goal,
  title   = {Global self-sustaining and local inheritance for source-free unsupervised domain adaptation},
  author  = {Peng, Lin and He, Yuhang and Wang, Shaokun and Song, Xiang and Dong, Songlin and Wei, Xing and Gong, Yihong},
  journal = {Pattern Recognition},
  volume  = {155},
  pages   = {110679},
  year    = {2024},
  doi     = {10.1016/j.patcog.2024.110679}
}
```

## License

The code in this repository is released under the [MIT License](LICENSE). The paper, datasets, dependencies, and externally downloaded weights retain their respective licenses and terms.

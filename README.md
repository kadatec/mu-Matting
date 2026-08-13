Repository for **αMatte4K & µMatting: Dataset and Model for Ultra-Micro
Precision Alpha Video Matting**.

## Overview

High-resolution human video matting must recover accurate alpha values in
semi-transparent and detail-rich regions while preserving temporal consistency
and practical efficiency. µMatting is a resolution-agnostic two-stage framework
designed for this setting:

1. **Coarse Alpha Predictor (CAP)** uses a portrait-aware masked autoencoder
   to localize the subject and estimate a coarse alpha matte.
2. **Fractional Alpha Refiner (FAR)** refines critical regions such as hair,
   clothing boundaries, and translucent areas with sparse 3D convolutions over
   neighboring frames.

The paper also introduces αMatte4K, a large-scale 4K human
video matting dataset generated with physically based rendering (PBR), which
provides accurate alpha annotations and physically coherent
foreground-background compositions. Together, αMatte4K and µMatting target
accurate, temporally stable, and efficient matting for high-resolution video.

## 1. Prepare the environment

The compatible stack uses Python 3.10, PyTorch 2.0.0+cu118, CUDA 11.8,
TorchVision 0.15.1+cu118, MMEngine 1.0.0, MMSegmentation 1.0.0, OpenCV 4.10,
and spconv 1.2.1. Other combinations may work, but recent spconv releases have
a different API.

Clone the official Sapiens repository and install its environment by following
its instructions:

```bash
git clone https://github.com/facebookresearch/sapiens.git third_party/sapiens
```

Download the official Sapiens 0.3B checkpoint separately. Its filename must be:

```text
sapiens_0.3b_epoch_1600_clean.pth
```

Install a CUDA/PyTorch-compatible spconv, then install both packages.
The same environment can be used for both phases:

```bash
conda create -n matting python=3.10
conda activate matting

export PYTHONPATH="CAP/src:third_party/sapiens/seg:third_party/sapiens/pretrain:third_party/sapiens/engine:third_party/sapiens/cv${PYTHONPATH:+:${PYTHONPATH}}"

python -m pip install -e CAP --no-deps
python -m pip install -e FAR
```

The CAP package uses `--no-deps` so pip does not replace the compatible
OpenMMLab packages in the environment. If the selected checkout requires
optional packages such as
`ftfy` or `easing_functions`, install the matching packages required by that
checkout before continuing.

## 2. Prepare CAP data

Place the three training datasets under one parent directory:

```text
CAP/data/
  HHM50K_img_alpha/
    images/
    alphas/
  RVM_img_alpha/
    images/
    alphas/
  MatteHuman_img_alpha/
    images/
    alphas/
  validation_img_alpha/
    images/
    alphas/
```

## 3. Train CAP

Run from the repository root:

```bash
export SAPIENS_ROOT="third_party/sapiens"
export SAPIENS_PRETRAINED_CHECKPOINT="weights/sapiens_0.3b_epoch_1600_clean.pth"

bash CAP/scripts/train.sh \
  8 \
  work_dirs/coarse_alpha
```

Place the official checkpoint at `CAP/weights/`. Training data defaults to
`CAP/data/`.

CAP trains for 30 epochs with batch size 4 and validates after every epoch.

Copy the trained checkpoint to the path used by FAR:

```bash
mkdir -p checkpoints
cp CAP/work_dirs/coarse_alpha/epoch_30.pth \
   checkpoints/cap_final.pth
```

## 4. Prepare FAR data

FAR training uses RVM and MatteHuman:

```text
FAR/data/
  RVM_img_alpha/
    fgr/<clip>/<frame>
    pha/<clip>/<frame>
  BackgroundVideos/
    <clip>/<frame>
  MatteHuman_video_alpha/
    img/<clip>/<frame>
    pha/<clip>/<frame>
    fgr/<clip>/<frame>
    bgr/<clip>/<frame>
```

RVM foreground and alpha frames are combined online with a randomly selected
background video. MatteHuman directly supplies its existing composite,
foreground, alpha, and background frames.

Validation and evaluation use separate JSON manifests. Every clip contains
exactly four frame records. Each record requires `image` and `alpha`;
`foreground` and `background` are optional. See the FAR README for the complete
JSON example.

## 5. Configure FAR

Copy the example configuration and edit its paths.

```bash
cp FAR/configs/example.toml \
   FAR/configs/local.toml
```

A complete configuration looks like:

```toml
[model]
first_stage_config = "../../CAP/configs/sapiens_0.3b_alpha_model.py"
first_stage_checkpoint = "../../checkpoints/cap_final.pth"
dilation_kernel = 5

[data]
sequence_length = 4
rvm_root = "../data/RVM_img_alpha"
background_root = "../data/BackgroundVideos"
mattehuman_root = "../data/MatteHuman_video_alpha"
validation_manifest = "../data/validation.json"
evaluation_manifest = "../data/evaluation.json"
rvm_repeat = 1
mattehuman_repeat = 2
image_size = [512, 512]
lowres_size = [512, 512]

[loss]
alpha_weights = [0.1, 0.1, 1.0]
composition_weight = 0.5
coherence_weight = 0.0
laplacian_levels = 3

[train]
batch_size = 32
num_workers = 32
epochs = 30
learning_rate = 1e-4
min_learning_rate = 1e-5
betas = [0.9, 0.999]
decay_every = 5
validation_interval = 1
seed = 0

[validation]
batch_size = 1
num_workers = 1

[evaluation]
batch_size = 1
num_workers = 1

[output]
best_checkpoint = "../outputs/best_joint.pt"
final_checkpoint = "../outputs/final_joint.pt"
```

The FAR config must reference `sapiens_0.3b_alpha_model.py`, not the CAP
training config.

## 6. Train FAR

```bash
python -m sparsemat3d.train \
  --config FAR/configs/local.toml \
  --device cuda
```

CAP is loaded once, frozen, and kept in evaluation mode. Only FAR is passed to
the optimizer. Validation runs after every epoch without updating parameters.
The checkpoint with the lowest validation MAD is written to `best_checkpoint`,
and the last epoch is written to `final_checkpoint`.

## 7. Evaluate and run inference

Evaluation reports MAD, MSE, Grad, and dtSSD. Evaluate the configured manifest
and final joint checkpoint:

```bash
python -m sparsemat3d.evaluate \
  --config FAR/configs/local.toml \
  --checkpoint FAR/outputs/final_joint.pt \
  --device cuda
```

Infer a local video. Frames are processed in non-overlapping groups of four;
the final group is padded by repeating its last frame:

```bash
python -m sparsemat3d.infer \
  --config FAR/configs/local.toml \
  --checkpoint FAR/outputs/final_joint.pt \
  --video inputs/input.mp4 \
  --output-dir outputs/alpha_frames \
  --device cuda
```

Video outputs are named `alpha_000000.png`, `alpha_000001.png`, and so on.

See each phase README for package-level details.

## Acknowledgements and licenses

We thank the authors of
[Sapiens](https://github.com/facebookresearch/sapiens) and
[SparseMat](https://github.com/nowsyn/SparseMat) for making their work
available. Sapiens, SparseMat, spconv, PyTorch, and other external
dependencies are used under their respective licenses. Cite the corresponding
papers when using this repository.

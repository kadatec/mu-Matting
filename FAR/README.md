# FAR

FAR implements 3D sparse video alpha refinement, including model, data loading,
training, evaluation, and inference.

## Architecture and training

- `model.first_stage_config` and `model.first_stage_checkpoint` initialize the
  CAP model once at the start of training.
- CAP stays in evaluation mode and has `requires_grad=False`; the optimizer
  receives only FAR parameters.
- `SparseVideoResNet18` and `SHMVideo` use random initialization.
- Training writes one final joint checkpoint containing both stages.
  Evaluation and inference load this complete checkpoint.

## Installation

Activate an environment containing the CAP dependencies, CUDA-compatible
PyTorch, and a compatible `spconv`. Install both packages from the repository
root:

```bash
python -m pip install -e CAP --no-deps
python -m pip install -e FAR
cd FAR
```

Model execution also requires MMEngine, MMSegmentation, and MMPReTrain. Install
an `spconv` wheel compatible with the local CUDA/PyTorch stack. The compatible
environment uses Python 3.10, PyTorch 2.0.0+cu118, CUDA 11.8, MMEngine 1.0.0,
MMSegmentation 1.0.0, and spconv 1.2.1.

## Configuration

Copy `configs/example.toml` to a local file and edit its paths:

```bash
cp configs/example.toml configs/local.toml
```

- `[model]` supplies
  `CAP/configs/sapiens_0.3b_alpha_model.py`, the completed CAP
  checkpoint, and an odd positive `dilation_kernel`.
- `[data]` fixes `sequence_length = 4`, points to the three training roots and
  separate validation/evaluation manifests, and defines `(height, width)` as
  `image_size` and `lowres_size`.
- `[loss]` configures the three alpha-scale weights and optional composition,
  temporal-coherence, and Laplacian losses.
- `[train]` configures batch size, workers, epochs, optimizer, learning-rate
  decay, validation interval, and random seed.
- `[validation]` configures validation loading during training.
- `[evaluation]` configures evaluation loading.
- `[output]` names the best and final joint checkpoints. Evaluation and video
  inference use the final checkpoint when `--checkpoint` is omitted.

## Tensor contract

Data loaders expose one consistent batch interface:

- `images`: normalized RGB tensors with shape `B, 3, T, H, W`; training uses
  the sampled high-resolution patch.
- `lowres_images`: 8-bit BGR tensors with shape `B, T, 3, h, w` for CAP.
  Training creates them from the complete composite frames before
  spatial cropping.
- `crop_boxes`: training-only normalized coordinates with shape `B, T, 4` in
  `x1, y1, x2, y2` order. Every frame in a sampled clip receives the same box.
- `alphas`: target alpha tensors with shape `B, 1, T, H, W`.
- `foregrounds` and `backgrounds`: normalized RGB tensors matching `images`
  when composition sources are available.

The model returns `multiscale_alphas`, `refined_alphas`, `coarse_alphas`, and
`unknown_region_mask`. Training losses consume the same descriptive batch and
model-output names. CAP processes each complete
`lowres_images` frame. During training, its coarse alpha is cropped with
`crop_boxes` and resized to the high-resolution patch; evaluation and inference
without `crop_boxes` resize the complete coarse alpha directly.

## Training datasets

Training uses two dataset readers with distinct layouts. The RVM root has
paired foreground and alpha clips:

```text
data/RVM_img_alpha/
├── fgr/<clip>/<frame>
└── pha/<clip>/<frame>
```

Background videos are separate:

```text
data/BackgroundVideos/<clip>/<frame>
```

RVM foreground and alpha frames are composited online with background videos.

MatteHuman stores all four modalities under one root:

```text
data/MatteHuman_video_alpha/
├── img/<clip>/<frame>
├── pha/<clip>/<frame>
├── fgr/<clip>/<frame>
└── bgr/<clip>/<frame>
```

Its existing composites are read from `img`; `fgr` and `bgr` are retained for
composition loss. It does not receive the RVM-specific augmentation.

## Validation and evaluation manifests

Both manifests use the same format: a JSON list of clips or an object with a
`clips` list. Every clip must contain exactly four frame objects:

```json
{
  "clips": [
    {
      "frames": [
        {
          "image": "rgb/0000.png",
          "alpha": "alpha/0000.png",
          "foreground": "fg/0000.png",
          "background": "bg/0000.png"
        },
        {
          "image": "rgb/0001.png",
          "alpha": "alpha/0001.png",
          "foreground": "fg/0001.png",
          "background": "bg/0001.png"
        },
        {
          "image": "rgb/0002.png",
          "alpha": "alpha/0002.png",
          "foreground": "fg/0002.png",
          "background": "bg/0002.png"
        },
        {
          "image": "rgb/0003.png",
          "alpha": "alpha/0003.png",
          "foreground": "fg/0003.png",
          "background": "bg/0003.png"
        }
      ]
    }
  ]
}
```

Each manifest requires `image` and `alpha`; `foreground` and `background` are
optional unless the caller explicitly requests composition fields.

## Training

```bash
python -m sparsemat3d.train \
  --config configs/local.toml \
  --device cuda
```

Training uses FAR with frozen CAP. Validation runs after every epoch without
gradient updates. The lowest validation MAD determines the best joint
checkpoint, while the final epoch is saved separately.

## Evaluation

Evaluation reports MAD, MSE, Grad, and dtSSD.

Use the checkpoint configured in `[output]`:

```bash
python -m sparsemat3d.evaluate \
  --config configs/local.toml \
  --device cuda
```

Or select another final joint checkpoint:

```bash
python -m sparsemat3d.evaluate \
  --config configs/local.toml \
  --checkpoint outputs/final_joint.pt \
  --device cuda
```

## Video inference

Video frames are processed as non-overlapping four-frame clips. The final clip
is padded by repeating its last frame, and only real frames are written:

```bash
python -m sparsemat3d.infer \
  --config configs/local.toml \
  --checkpoint outputs/final_joint.pt \
  --video inputs/example.mp4 \
  --output-dir outputs/example_alpha \
  --device cuda
```

Outputs are named `alpha_000000.png`, `alpha_000001.png`, and so on.

## Package layout

- `sparse_video_resnet.py` and `shm_video.py`: 3D sparse backbone and
  hierarchical decoder.
- `model.py`: frozen CAP adapter and CAP-FAR wrapper.
- `data.py`: four-frame manifest and video data helpers.
- `losses.py` and `metrics.py`: training objectives and evaluation metrics.
- `train.py`, `evaluate.py`, and `infer.py`: separate CLIs.
- `checkpoint.py`: strict final joint checkpoint format.

## Acknowledgement

We thank the authors of
[SparseMat](https://github.com/nowsyn/SparseMat) for making their work
available. Cite the SparseMat paper when using this repository.

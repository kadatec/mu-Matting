# CAP

CAP localizes the foreground subject and predicts a coarse alpha matte at
reduced resolution. It includes the dataset, transforms, model head, loss,
metric, configuration, and training launcher.

The encoder is initialized from the Sapiens-0.3B pretrained backbone. Training
uses `mmseg`, `mmpretrain`, `mmengine`, `mmcv`, PyTorch, OpenCV, and NumPy.

## Install

Activate an environment in which the required segmentation and pretraining
modules are importable, then install CAP:

```bash
cd CAP
python -m pip install -e . --no-deps
```

`--no-deps` preserves the compatible OpenMMLab packages in the environment.

## Data

Training uses these three roots:

```text
data/
  HHM50K_img_alpha/
  RVM_img_alpha/
  MatteHuman_img_alpha/
  validation_img_alpha/
```

Set `SAPIENS_ALPHA_DATA_ROOT` to relocate the parent directory and
`SAPIENS_ALPHA_VAL_ROOT` to select another validation root.

Each dataset contains paired image and alpha directories.

## Train

From this directory, configure the source checkout and backbone checkpoint:

```bash
export SAPIENS_ROOT="../third_party/sapiens"
export SAPIENS_PRETRAINED_CHECKPOINT="weights/sapiens_0.3b_epoch_1600_clean.pth"

bash scripts/train.sh 8 "work_dirs/coarse_alpha"
```

The default data parent is `data/`, and the default validation root is
`data/validation_img_alpha/`.

The backbone is initialized with
`sapiens_0.3b_epoch_1600_clean.pth`.

Training runs for 30 epochs with batch size 4 and four data workers. AdamW uses
a learning rate of `5e-4`, betas `(0.9, 0.999)`, weight decay `0.1`, and layer
decay `0.85`. Validation runs after every epoch with batch size 4 and four data
workers; it computes metrics without updating model parameters.

## FAR integration

FAR loads `configs/sapiens_0.3b_alpha_model.py` with the trained CAP
checkpoint.

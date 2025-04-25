# Dependencies
Install the required dependencies(Python 3.13+ recommended):
```bash
pip install -r requirements.txt
```
This codebase is built on PyTorch 2.6 and CUDA 12.4—please make sure your setup matches these versions for full compatibility.

# Load the Multi-Task Model and Dataloaders

Clone this repository into your project folder, and import load_mt_model and load_dataloaders from the util_simulator module.

```python
from util_simulator import load_mt_model, load_dataloaders

# Load a multitask model with EfficientNet-B2 as the backbone.
# This model includes task-specific heads for:
# - Classification (trained on EuroSAT)
# - Semantic Segmentation (trained on LandCover.ai)

# The model's forward method supports:
# 1. A specific task (e.g., 'classification' or 'segmentation'):
#    → Returns output from that task-specific head.
# 2. No task specified:
#    → Returns a dictionary of outputs from all task heads.
model = load_mt_model(option='b2')

# Load a dictionary of dataloaders:
# Keys: 'classification_train', 'classification_val', 
#       'segmentation_train', 'segmentation_val'
# The classification dataloaders are from the EuroSAT dataset,
# The segmentation dataloaders are from the LandCover.ai dataset.
dataloader_set = load_dataloaders()
```

# Run Experiments

## How to run the code
Use the following command patterns to run experiments:

| Option | Example |
|---------|---------|
| Change backbone | `python main.py model.backbone=b0` (b0 to b7 supported) |
| Freeze first 6 layers | `python main.py freeze=none` (options : none, layer6, layer7.0, layer7.1(for b5, b6, b7), layer7.2(for b7), layer7, backbone) |
| Set up the model and dataset | `python main.py model=deeplabv3 dataset=segmentation/deepglobelandcover` |

You can also find the default configuration in `configs/default.yaml`. Please modify the following options to suit your run:

```yaml
# configs/default.yaml

freeze: full # none | layer6 | layer7.0 | layer7.1(b5, b6, b7) | layer7.2(b7) | layer7 | backbone

# Optimizer configuration (shared across experiments)
optimizer:
  _target_: torch.optim.AdamW
  lr: 1e-3
  weight_decay: 0.01
  eps: 1e-8

# Scheduler configuration (shared across experiments)
scheduler:
  _target_: torch.optim.lr_scheduler.StepLR
  step_size: 10
  gamma: 0.1

trainer:
  epochs: 40 
  patience: 40
  batch_size: 16 
```

## Directory Layout

```
configs/
 ├─ default.yaml            # root Hydra config
 ├─ model/                  # model presets
 └─ dataset/                # dataset presets
src/                        # project code
train.py                    # training entry‑point
evaluate.py                 # validation / test script
infer.py                    # single‑image inference
requirements.txt            # Python deps
```
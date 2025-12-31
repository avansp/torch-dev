<div align="center">

# TORCH-DEV

[![python](https://img.shields.io/badge/-Python_3.12-blue?logo=python&logoColor=white)]()
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/docs/pytorch/stable/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

</div>

This is a template to create a pyTorch structure using [Lightning](https://github.com/Lightning-AI/pytorch-lightning) and [Hydra](https://github.com/facebookresearch/hydra) framework, simplified from [Ashleve's Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template/tree/main) template.

<div align="center">

[![Button Icon]](https://github.com/ashleve/lightning-hydra-template/generate)

</div>

## Installation

```{bash}
pip install -r requirements.txt
```

Install the package script (note use `-e` for development)
```{bash}
pip install .
```

## Training

An example of MNIST classification is provided for training. This is defined in the [mnist.yaml](src/configs/task/mnist.yaml) task config file.

```{bash}
train task=mnist
```

<details>
<summary><b>Show data config</b></summary>

MNIST data is defined by [MNISTDataModule](src/modules/data/mnist_datamodule.py) class, and instantiated using a config file in [mnist.yaml](src/configs/task/mnist.yaml) file, as follows:

```yaml
data:
  _target_: src.modules.data.mnist_datamodule.MNISTDataModule
  data_dir: ${paths.data_dir}
  batch_size: 128 # Needs to be divisible by the number of devices (e.g., if in a distributed setup)
  train_val_test_split: [55_000, 5_000, 10_000]
  num_workers: 0
  pin_memory: False
```

</details>

<details>
<summary><b>Show console output</b></summary>

```console
[2025-12-29 08:33:37,155][root][INFO] - Train cfg.name='mnist' started.
...
[2025-12-29 08:35:58,100][root][INFO] - Saving last checkpoint to ep0016_vl0.0996_tl0.0139.ckpt
[2025-12-29 08:35:58,108][root][INFO] - Best checkpoint path: ep0005_vl0.0760_tl0.0372.ckpt
[2025-12-29 08:35:58,110][root][INFO] - Last train result = {'val/loss': tensor(0.0996), 'val/acc': tensor(0.9764), 'val/acc_best': tensor(0.9792), 'train/loss': tensor(0.0139), 'train/acc': tensor(0.9953)}
[2025-12-29 08:35:58,110][root][INFO] - Train cfg.name='mnist' terminated, elapsed 143.694471932 sec.
[2025-12-29 08:35:58,111][root][INFO] - Output dir: /data/avan/dev/torch-dev/outputs/mnist/runs/2025-12-29_08-33-37
```

</details>

<details>
<summary><b>Show output directory</b></summary>

```
output directory
├── checkpoints           <- Model checkpoints
│   ├── best_model.ckpt   <- Link to the best model file
│   ├── last.ckpt         <- Link to the last model file
│   ├── ....ckpt          <- Saved model file
│   └── ....ckpt
│
├── config_tree.log
├── config.yaml
├── console.log
│
├── csv                   <- Logs from CSV callback
│   └── version_0
│       ├── hparams.yaml
│       └── metrics.csv
└── tensorboard           <- Logs from Tensorboard
    └── version_0
        ├── ...
        └── ...
```

</details>

## How to ... ?

<details>
<summary><b>How to check configurations without training ?</b></summary>

Activate the `dry_run` config parameter.
```console
train task=mnist dry_run=true
```

</details>

[Button Icon]: https://img.shields.io/badge/USE_THIS_TEMPLATE-orange?style=for-the-badge

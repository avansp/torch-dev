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

## MNIST classification

```{bash}
train task=mnist
```

<details>
<summary><b>Show data config</b></summary>

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


[Button Icon]: https://img.shields.io/badge/USE_THIS_TEMPLATE-orange?style=for-the-badge

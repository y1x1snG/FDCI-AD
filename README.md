# FDCI-AD: Full-Dimensional Collaborative Interpretable Network for Anomaly Detection

**Built upon [GLASS](https://arxiv.org/abs/2407.09359)¹** — a unified anomaly synthesis strategy with gradient ascent for industrial anomaly detection and localization.

_FDCI-AD extends GLASS by introducing three plug-in modules — **FDSIA**, **MPDC**, and **EDCT** — that enrich the global and local anomaly synthesis branches of the GLASS framework, improving detection of weak and structurally complex defects._

¹ Q. Chen, H. Luo, C. Lv, and Z. Zhang, “A Unified Anomaly Synthesis Strategy with Gradient Ascent for Industrial Anomaly Detection and Localization,” arXiv:2407.09359, 2024.

![Overall architecture of FDCI-AD](figures/fdci_ad_framework.png)

_Figure: Overall architecture of FDCI-AD. The underlying GLASS framework (Normal / GAS / LAS branches) is extended with the proposed FDSIA, MPDC, and EDCT modules._

## Introduction

FDCI-AD is a unified framework for **unsupervised industrial anomaly detection and localization**. It is built on top of **GLASS** (Global and Local Anomaly Co-Synthesis Strategy)¹ and contributes three modules:

- **FDSIA** (Full-Dimensional Sequential Interaction Attention) — a sequential interaction attention integrated into the GLASS global branch to capture full-dimensional feature dependencies (§3.1).
- **MPDC** (Multi-Path Depthwise Convolution) — a multi-path depthwise convolution (e.g., large-kernel ConvNeXt style) that expands the receptive field and enriches the semantic information extraction of the GLASS global branch (§3.2).
- **EDCT** (Explainable Dynamic Channel Transformation) — a dynamic channel transformation that enhances feature representation and interpretability (§3.3).

These modules are integrated into the GLASS global and local anomaly synthesis branches and trained end-to-end together with the original GLASS objective.

## Environment

```bash
conda create -n fdci_ad python=3.9.15
conda activate fdci_ad
pip install -r requirements.txt
```

The experiments in the paper were conducted on an NVIDIA Titan V GPU; similar hardware and package versions are recommended.

## Data Preparation

The released code is evaluated on **MVTec AD** and **VisA**. **DTD** is used as an auxiliary texture dataset for data augmentation (same role as in GLASS).

- MVTec AD: https://www.mvtec.com/company/research/datasets/mvtec-ad/
- VisA: https://github.com/amazon-science/spot-diff/
- DTD: https://www.robots.ox.ac.uk/~vgg/data/dtd/

Point `datapath` (the evaluation dataset) and `augpath` (the DTD images) in the run scripts to your local paths.

## Run Experiments

Edit the dataset path, augmentation path, and class list in `shell/run-mvtec.sh` or `shell/run-visa.sh`, then launch from the repository root:

```bash
bash shell/run-mvtec.sh
# or
bash shell/run-visa.sh
```

Each script changes into the repository root and runs `python main.py` with the GLASS/FDCI-AD backbone and dataset arguments, so that relative paths (`datasets/excel/`, `results/`) resolve correctly regardless of where you invoke it from. `--test ckpt` **trains** the model (and skips training if a checkpoint already exists) before evaluating it; `--test train` **evaluates an existing checkpoint only**. Other hyperparameters (`--lr`, `--noise`, `--radius`, `--meta_epochs`, etc.) can be adjusted in the same script.

## Project Structure

- `main.py`, `glass.py`, `model.py`, `common.py`, `backbones.py`, `resnet12.py` — the FDCI-AD pipeline and the modified WideResNet-50 backbone.
- `FDSIA.py`, `MPDC.py`, `EDCT.py` — the three proposed modules (§3.1–3.3).
- `metrics.py`, `loss.py`, `perlin.py`, `utils.py`, `datasets/` — supporting utilities and dataset loaders (MVTec AD, VisA).
- `torchvision_vendored/` — **vendored internal modules of PyTorch's `torchvision`** v0.15.1 (BSD 3-Clause, Copyright (c) Soumith Chintala 2016) that the backbone depends on; they are **not** part of FDCI-AD's own contribution. The full license text is reproduced in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
- `shell/run-mvtec.sh`, `shell/run-visa.sh` — launch scripts for the two evaluated datasets.

## Citation

If you use FDCI-AD, please cite this work and the base method it extends:

```bibtex
@article{wang2026fdciad,
  title  = {Full-Dimensional Collaborative Interpretable Network for Anomaly Detection},
  author = {Wang, Guogang and Liu, Mingpeng and Zhao, Xin and Chen, Yaqiu},
  journal = {iScience},
  year   = {2026},
  note   = {under review}
}
```

```bibtex
@article{chen2024unified,
  title  = {A Unified Anomaly Synthesis Strategy with Gradient Ascent for Industrial Anomaly Detection and Localization},
  author = {Chen, Qiyu and Luo, Huiyuan and Lv, Chengkan and Zhang, Zhengtao},
  journal = {arXiv preprint arXiv:2407.09359},
  year   = {2024}
}
```

## Acknowledgements

Thanks for the great inspiration from [GLASS](https://arxiv.org/abs/2407.09359) and [SimpleNet](https://github.com/DonaldRR/SimpleNet/).

## License

Released under the [MIT License](LICENSE). This repository is a derivative work of GLASS and inherits its MIT license; the full text of every third-party license is reproduced in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

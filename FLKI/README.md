# Project Name
Federated Learning under Knowledge Isolation (FLKI)

# Project Description
This project implements federated learning with knowledge distillation across multiple clients. It supports adaptive layer selection and Optimal Transport (OT)-based personalized weight updates. Supported datasets include `fmnist`, `cifar10`, and `tinyimagenet`.

# Installation Guide
## Dependencies
Use Python 3.8+ with a Conda environment:

```bash
conda create -n pytorch_env python=3.9 -y
conda activate pytorch_env
pip install torch torchvision numpy pandas matplotlib
```

## Configuration
- Data directory layout:
  - CIFAR-10: `data/cifar10/cifar-10-batches-py/`
  - FashionMNIST: `data/FashionMNIST/raw/`
  - TinyImageNet-200: `data/tiny-imagenet-200/`
- Optional config file: pass `--config` to load key-value pairs that are merged into runtime arguments.

# Usage
Main entry: see [Fedkd.py](Fedkd.py)

Common flags:
- `--dataset`: dataset name (`fmnist`, `cifar10`, `tinyimagenet`)
- `--num_classes`: number of classes (10 for CIFAR-10, 200 for TinyImageNet)
- `--trainer_num`: number of clients/trainers
- `--round`: communication rounds
- `--bs`: batch size
- `--lr`, `--momentum`: optimizer settings
- `--non_iid_degree`: Dirichlet alpha (data heterogeneity)
- `--layer_coverage_tau`: adaptive layer coverage threshold

Examples:
- CIFAR-10
```bash
python Fedkd.py --dataset cifar10 --num_classes 10 \
  --trainer_num 8 --round 100 --bs 64 --lr 0.01 --momentum 0.5 \
  --non_iid_degree 1.0 --layer_coverage_tau 0.8
```

- TinyImageNet
```bash
python Fedkd.py --dataset tinyimagenet --num_classes 200 \
  --trainer_num 8 --round 200 --bs 64 --lr 0.005 --momentum 0.9 \
  --non_iid_degree 1.0 --layer_coverage_tau 0.8
```

Outputs:
- Console shows per-round client accuracy and global accuracy
- Accuracy logs saved to `Intermediate_data/accuracy_{dataset}_{non_iid_degree}.xlsx`
- A plot window displays global accuracy vs. rounds

# Features
- Adaptive layer selection based on distinguishability
- Soft/Hard label distillation support
- OT-based personalization for selected layers
- Multi-dataset support: LeNet, ResNet18, ViT-B16
- Logging and training curve visualization


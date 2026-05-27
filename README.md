# ReLU Pruning for Privacy-Preserving Inference

## Overview

This repository implements a progressive ReLU pruning method based on Taylor importance estimation for privacy-preserving neural network inference. The approach reduces the number of ReLU activations in deep neural networks while maintaining model accuracy, which is crucial for efficient secure multi-party computation (MPC) and homomorphic encryption (HE) based inference.

## Key Features

- **Taylor Importance Estimation**: Uses first and second-order Taylor expansion to estimate the importance of ReLU activations
- **Progressive Pruning**: Gradually prunes ReLUs during training to maintain model performance
- **Layer-wise Importance**: Considers layer depth and gradient information for better pruning decisions
- **Knowledge Distillation**: Leverages teacher-student framework to preserve accuracy

## Requirements

```bash
pip install torch torchvision numpy einops
```

## Project Structure

```
.
├── train_teacher.py                              # Train teacher model
├── train_with_taylor_pruning_global_relu_pruning_abs.py  # Main training script with Taylor pruning
├── validate.py                                   # Validation script
├── models/                                       # Model architectures
├── helper/                                       # Training utilities
├── distiller_zoo/                                # Knowledge distillation modules
└── dataset/                                      # Dataset loaders
```

## Usage

### 1. Train Teacher Model

First, train a standard teacher model with full ReLU activations:

```bash
# CIFAR-10
python train_teacher.py --model='ResNet18' --dataset='cifar10'

# CIFAR-100
python train_teacher.py --model='ResNet18' --dataset='cifar100'

# Tiny ImageNet
python train_teacher.py --model='ResNet18' --dataset='tiny_imagenet' --batch_size=32
```

### 2. Train with Taylor-based ReLU Pruning

Apply progressive ReLU pruning using Taylor importance estimation:

```bash
# CIFAR-100 with 5% ReLU retention (95% pruning)
python train_with_taylor_pruning_global_relu_pruning_abs.py \
    --path_t='save/models/ResNet18_cifar100_batch128/ResNet18_best.pth' \
    --alpha=0.3 \
    --gamma=0.01 \
    --model_s='CustomResNet18' \
    --dataset='cifar100' \
    --distill='kd' \
    --global_keep_ratio=0.05

# CIFAR-100 with 10% ReLU retention
python train_with_taylor_pruning_global_relu_pruning_abs.py \
    --path_t='save/models/ResNet18_cifar100_batch128/ResNet18_best.pth' \
    --alpha=0.5 \
    --gamma=0.5 \
    --model_s='CustomResNet18' \
    --dataset='cifar100' \
    --distill='kd' \
    --global_keep_ratio=0.1
```

### 3. Validate Model

```bash
python validate.py \
    --model='CustomResNet18' \
    --dataset='cifar100' \
    --path='save/student_model/CustomResNet18_pruned.pth'
```

## Key Parameters

- `--path_t`: Path to pre-trained teacher model
- `--model_s`: Student model architecture (e.g., CustomResNet18)
- `--dataset`: Dataset name (cifar10, cifar100, tiny_imagenet)
- `--global_keep_ratio`: Ratio of ReLUs to keep (e.g., 0.05 = 5% ReLUs retained)
- `--alpha`: Weight for knowledge distillation loss
- `--gamma`: Weight for task loss
- `--distill`: Distillation method (kd for knowledge distillation)

## Method Overview

The pruning method consists of:

1. **Importance Calculation**: Compute Taylor importance scores for each ReLU activation
   - First-order: gradient magnitude
   - Second-order: Hessian diagonal approximation
   - Layer-wise weighting based on depth

2. **Progressive Pruning**: Gradually increase pruning ratio during training
   - Start with low pruning ratio
   - Progressively increase to target ratio
   - Allow model to adapt at each pruning step

3. **Knowledge Distillation**: Maintain accuracy using teacher guidance
   - Soft label matching
   - Feature-level distillation

## Results

The method achieves significant ReLU reduction (90-95%) while maintaining competitive accuracy on CIFAR-10, CIFAR-100, and Tiny ImageNet datasets.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{li2026relupruner,
  title={ReLUPruner: Rethinking ReLU Importance with Taylor Expansion for Efficient Private Inference},
  author={Li, Zhenpeng and Liu, Jinshuo and Wang, Xinyan and Wang, Lina and Pan, Jeff Z},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={28},
  pages={23328--23336},
  year={2026}
}
```

## License

MIT License

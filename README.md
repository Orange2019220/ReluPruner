# ReLUPruner

Official implementation of **ReLUPruner: Rethinking ReLU Importance with Taylor Expansion for Efficient Private Inference** (AAAI 2026).

ReLUPruner estimates the importance of spatial ReLU positions with first- and second-order Taylor expansion and progressively prunes them under a global ReLU budget.

## Environment Setup

Python 3.9 or later is recommended.

```bash
conda create -n relupruner python=3.9
conda activate relupruner
```

Install a PyTorch build compatible with your CUDA version, then install this repository:

```bash
# Example for CUDA 12.8
pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128

pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

## Datasets

The following datasets are supported:

- CIFAR-10
- CIFAR-100
- Tiny-ImageNet

By default, the configuration files use `data/` as the dataset root. You can either edit `DATA_ROOT` in the corresponding file under `configs/` or override it when launching an experiment:

```bash
DATA_ROOT=/path/to/datasets bash cifar10.sh
```

Expected CIFAR layout:

```text
data/
  cifar-10-batches-py/
  cifar-100-python/
```

Expected Tiny-ImageNet layout:

```text
data/
  tiny-imagenet-200/
    train/<class>/images/*.JPEG
    val/images/*.JPEG
    val/val_annotations.txt
```

The class-organized validation layout `val/images/<class>/*.JPEG` is also supported.

## Supported Models

| Dataset | ResNet-18 | ResNet-34 | WRN-22-8 |
|---|:---:|:---:|:---:|
| CIFAR-10 | ✓ | ✓ | ✓ |
| CIFAR-100 | ✓ | ✓ | ✓ |
| Tiny-ImageNet | ✓ | ✓ | — |

## Configuration

Full experiment configurations are provided in:

```text
configs/cifar10_resnet18.env
configs/cifar100_resnet18.env
configs/tinyimagenet_resnet18.env
```

The main fields to modify are:

```bash
PYTHON_BIN=${PYTHON_BIN:-python}
DATA_ROOT=${DATA_ROOT:-data}
MODEL=resnet18

# Set exactly one ReLU budget.
TARGET_RELUS=49152
KEEP_RATIO=
```

Use `TARGET_RELUS` when comparing different architectures under the same absolute ReLU budget. Use `KEEP_RATIO` when each model should retain the same fraction of its own ReLU positions.

## Training

Run the complete teacher, pruning, finetuning, and evaluation pipeline with a dataset wrapper:

```bash
bash cifar10.sh
bash cifar100.sh
bash tinyimagenet.sh
```

All wrappers call the same entry point:

```bash
bash run_experiment.sh configs/cifar10_resnet18.env all
```

Individual stages can be executed separately:

```bash
bash cifar10.sh teacher
bash cifar10.sh prune
bash cifar10.sh finetune
bash cifar10.sh evaluate
```

Command-line options can override the model, budget, dataset path, batch size, and worker count:

```bash
bash cifar10.sh all \
  --model resnet34 \
  --target-relus 49152 \
  --batch-size 64 \
  --workers 4
```

To inspect the total number of maskable ReLU positions without training:

```bash
bash cifar10.sh info --model resnet34
```

Existing checkpoints are reused when `SKIP_COMPLETED_STAGES=1`. Teacher checkpoints are shared by experiments with the same dataset and model, so changing only the ReLU budget does not retrain the teacher. Use `--force` to rerun completed stages.

## Multiple Models and Budgets

Experiment matrices are defined in:

```text
configs/cifar10_matrix.env
configs/cifar100_matrix.env
configs/tinyimagenet_matrix.env
```

Preview a matrix without starting training:

```bash
bash run_matrix.sh configs/cifar10_matrix.env --dry-run
```

Run the configured matrix:

```bash
bash run_matrix.sh configs/cifar10_matrix.env
bash run_matrix.sh configs/cifar100_matrix.env
bash run_matrix.sh configs/tinyimagenet_matrix.env
```

Edit `MODELS`, `BUDGETS`, and `MODEL_BATCH_SIZES` in a matrix file to select the experiments to run. See [configs/README.md](configs/README.md) for all configuration fields.

## Entry Points

- `scripts.train_teacher`: train a dense teacher model.
- `scripts.train_pruner`: learn a global Taylor-based ReLU mask.
- `scripts.finetune`: finetune the student with a fixed mask.
- `scripts.evaluate`: evaluate a teacher or pruned checkpoint.
- `run_experiment.sh`: run one configuration or one training stage.
- `run_matrix.sh`: run multiple models and ReLU budgets.

## Outputs

By default, checkpoints and logs are stored under:

```text
outputs/<dataset>/
logs/<dataset>/
```

The final checkpoint for each experiment is saved as `final.pt`. The configuration used for the run is copied into the corresponding output directory.

## Citation

```bibtex
@inproceedings{li2026relupruner,
  title={ReLUPruner: Rethinking ReLU Importance with Taylor Expansion for Efficient Private Inference},
  author={Li, Zhenpeng and Liu, Jinshuo and Wang, Xinyan and Wang, Lina and Pan, Jeff Z.},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={28},
  pages={23328--23336},
  year={2026}
}
```

## License

This project is released under the [MIT License](LICENSE).

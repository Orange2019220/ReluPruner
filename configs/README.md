# Experiment Configuration

The `.env` files in this directory define dataset paths, model names, ReLU budgets, optimization settings, and output locations.

## Single Experiments

```bash
bash cifar10.sh all --model resnet18 --target-relus 49152
bash cifar100.sh all --model resnet34 --target-relus 49152
bash tinyimagenet.sh all --model resnet18 --target-relus 196608
```

The corresponding base configurations are:

- `cifar10_resnet18.env`
- `cifar100_resnet18.env`
- `tinyimagenet_resnet18.env`

## Environment and Data

The default values use the active Python environment and the repository-local `data/` directory:

```bash
PYTHON_BIN=${PYTHON_BIN:-python}
DATA_ROOT=${DATA_ROOT:-data}
```

Edit these values in a configuration file, or override them when running a command:

```bash
PYTHON_BIN=/path/to/python \
DATA_ROOT=/path/to/datasets \
bash cifar10.sh
```

The main runtime fields are:

```bash
DEVICE=cuda
CUDA_DEVICE=0
WORKERS=4
PIN_MEMORY=0
SEED=0
```

## ReLU Budget

Set exactly one of `TARGET_RELUS` and `KEEP_RATIO`:

```bash
TARGET_RELUS=49152
KEEP_RATIO=
```

or:

```bash
TARGET_RELUS=
KEEP_RATIO=0.10
```

`TARGET_RELUS` specifies an exact global ReLU count. `KEEP_RATIO` specifies a model-relative fraction.

## Training Settings

Each training stage has independent epoch, batch-size, and learning-rate fields:

```bash
TEACHER_EPOCHS=240
TEACHER_BATCH_SIZE=128
TEACHER_LR=0.05

PRUNER_EPOCHS=150
PRUNER_BATCH_SIZE=128
PRUNER_LR=0.05

FINETUNE_EPOCHS=240
FINETUNE_BATCH_SIZE=128
FINETUNE_LR=0.01
```

The following command overrides all stage batch sizes for one run:

```bash
bash cifar10.sh all --model wrn_22_8 --batch-size 32
```

## Multiple Models and Budgets

Matrix configurations contain the base configuration, selected models, budgets, and model-specific batch sizes:

```bash
BASE_CONFIG=configs/cifar10_resnet18.env
MODELS="resnet18 resnet34 wrn_22_8"
BUDGET_TYPE=target_relus
BUDGETS="49152 12779 6389"
MODEL_BATCH_SIZES="resnet18:128 resnet34:64 wrn_22_8:32"
```

Preview or run a matrix with:

```bash
bash run_matrix.sh configs/cifar10_matrix.env --dry-run
bash run_matrix.sh configs/cifar10_matrix.env
```

## Output Reuse

`SKIP_COMPLETED_STAGES=1` reuses existing checkpoints. Teacher checkpoints are keyed by dataset and model and are reused across different ReLU budgets.

Use `--force` only when all selected completed stages should be rerun.

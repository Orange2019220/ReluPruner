# 实验配置说明

日常使用只需要修改 `.env` 文件，然后运行一个 Bash 脚本。

## 配置文件

- `cifar10_resnet18.env`：CIFAR-10 完整训练基准配置；
- `cifar100_resnet18.env`：CIFAR-100 完整训练基准配置；
- `tinyimagenet_resnet18.env`：Tiny-ImageNet 完整训练基准配置；
- `cifar10_matrix.env`：CIFAR-10 多模型、多预算矩阵；
- `cifar100_matrix.env`：CIFAR-100 多模型、多预算矩阵；
- `tinyimagenet_matrix.env`：Tiny-ImageNet 的 ResNet18/34 多预算矩阵；
- `smoke_*.env`：只用于端到端短流程检查，不用于报告精度。

## 推荐用法

单个实验：

```bash
bash cifar10.sh all --model resnet18 --target-relus 49152
bash cifar10.sh all --model resnet34 --target-relus 49152 --batch-size 64
bash cifar10.sh all --model wrn_22_8 --target-relus 49152 --batch-size 32
```

批量实验：

```bash
bash run_matrix.sh configs/cifar10_matrix.env --dry-run
bash run_matrix.sh configs/cifar10_matrix.env
```

## 绝对预算与保留比例

配置中必须且只能启用一种预算：

```bash
TARGET_RELUS=49152
KEEP_RATIO=
```

或：

```bash
TARGET_RELUS=
KEEP_RATIO=0.10
```

跨模型公平比较应使用 `TARGET_RELUS`。相同的 `KEEP_RATIO` 会因模型总 ReLU 数不同而得到不同的绝对预算。

CIFAR 输入下的总量：

| 模型 | 可剪 ReLU 总量 |
|---|---:|
| ResNet18 | 491,520 |
| ResNet34 | 901,120 |
| WRN-22-8 | 1,392,640 |

ResNet18 常用比例与取整后的预算：

| `KEEP_RATIO` | ReLU 数量 |
|---:|---:|
| 0.10 | 49,152 |
| 0.026 | 12,779 |
| 0.013 | 6,389 |

如需严格复现实验表中的整数预算，直接填写 `TARGET_RELUS`，不要依赖浮点比例取整。

## Teacher 复用规则

teacher 文件默认保存为：

```text
outputs/<dataset>/<model>_teacher.pt
```

它只由数据集和模型决定，不由剪枝预算决定。连续运行同模型的多个预算时只训练一次 teacher；换模型后训练对应的新 teacher。`SKIP_COMPLETED_STAGES=1` 会复用已有 checkpoint。

若使用 `--force`，所有指定阶段都会重新执行并覆盖原结果。在矩阵训练中使用它会让每个预算重复训练 teacher，因此通常不要给矩阵命令添加 `--force`。

## 常改字段

```bash
DATA_ROOT=/home/zhenpengl/research-data/datasets
MODEL=resnet18
TARGET_RELUS=
KEEP_RATIO=0.10
TEACHER_BATCH_SIZE=128
PRUNER_BATCH_SIZE=128
FINETUNE_BATCH_SIZE=128
WORKERS=4
```

命令行中的 `--batch-size N` 会同时覆盖 teacher、剪枝、微调和评估的批大小。显存不足时建议先降低 `PRUNER_BATCH_SIZE`，因为 Taylor 重要性计算通常是显存峰值所在阶段。

矩阵文件主要修改：

```bash
MODELS="resnet18 resnet34 wrn_22_8"
BUDGET_TYPE=target_relus
BUDGETS="49152 12779 6389"
MODEL_BATCH_SIZES="resnet18:128 resnet34:64 wrn_22_8:32"
```

Tiny-ImageNet 不支持 `wrn_22_8`。

## 阶段命令

```bash
bash run_experiment.sh configs/cifar10_resnet18.env info
bash run_experiment.sh configs/cifar10_resnet18.env teacher
bash run_experiment.sh configs/cifar10_resnet18.env prune
bash run_experiment.sh configs/cifar10_resnet18.env finetune
bash run_experiment.sh configs/cifar10_resnet18.env evaluate
bash run_experiment.sh configs/cifar10_resnet18.env all
```

入口支持以下临时覆盖：

```text
--model NAME
--target-relus N
--keep-ratio R
--data-root PATH
--batch-size N
--workers N
--force
```

命令行覆盖模型或预算时，实验目录会自动命名，避免不同实验写入同一路径。直接编辑基础配置时，应同步修改 `EXPERIMENT_NAME`。

## 完整训练与 smoke 检查

完整训练时，下列字段必须保持为空：

```bash
TEACHER_MAX_TRAIN_BATCHES=
PRUNER_MAX_TRAIN_BATCHES=
FINETUNE_MAX_TRAIN_BATCHES=
MAX_VAL_BATCHES=
```

`smoke_*.env` 会把它们设为很小的数字，只用于确认数据、模型、掩码、损失和 checkpoint 链路能够运行。smoke 准确率没有实验意义。

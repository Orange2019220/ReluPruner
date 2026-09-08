# ReLUPruner

论文 **ReLUPruner: Rethinking ReLU Importance with Taylor Expansion for Efficient Private Inference**（AAAI 2026）的精简实现。

ReLUPruner 使用一阶与二阶 Taylor 展开估计各空间位置的 ReLU 重要性，在整个网络内进行渐进式全局剪枝。被保留的位置执行 ReLU，其余位置执行恒等映射，从而减少隐私推理中的非线性运算。

## 当前状态

仓库只保留论文自然图像实验所需的主流程：

1. 训练同数据集、同架构的稠密 teacher；
2. 训练 student，并根据全局 Taylor 分数逐步剪除 ReLU；
3. 固定掩码，使用分类、logit 蒸馏和 PRAM 损失微调；
4. 加载最终 checkpoint，报告准确率和实际 ReLU 预算。

本地已经完成以下检查：

- CIFAR-10 / ResNet18 完整训练结果：49,152 ReLU，测试准确率 94.17%；
- 自动测试全部通过；
- 下表中的 8 个数据集/模型组合均通过 teacher → 剪枝 → 微调 → 评估的 GPU 短流程检查。

短流程只验证代码兼容性，不代表完整训练精度。除上述 CIFAR-10/ResNet18 结果外，其余组合仍需按完整配置训练后才能与论文表格比较。

## 支持范围

| 数据集 | ResNet18 | ResNet34 | WRN-22-8 |
|---|:---:|:---:|:---:|
| CIFAR-10 | ✓ | ✓ | ✓ |
| CIFAR-100 | ✓ | ✓ | ✓ |
| Tiny-ImageNet | ✓ | ✓ | — |

WRN-22-8 只支持 32×32 输入；程序会在读取 Tiny-ImageNet 前拒绝该组合。Tiny-ImageNet 同时支持官方 `val/images + val_annotations.txt` 布局，以及 `val/images/<class>/` 的已整理布局。

## 安装

建议使用 Python 3.9 及以上版本和支持 CUDA 的 PyTorch：

```bash
python -m pip install -e ".[dev]"
```

RTX 50 系显卡可使用 PyTorch CUDA 12.8 构建。本仓库验证环境为：

```bash
python -m pip install --upgrade torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

## 数据集

CIFAR-10 和 CIFAR-100 的配置默认读取：

```text
/home/zhenpengl/research-data/datasets
```

Tiny-ImageNet 目录应为以下两种形式之一：

```text
<DATA_ROOT>/tiny-imagenet-200/train/<class>/images/*.JPEG
<DATA_ROOT>/tiny-imagenet-200/val/images/*.JPEG
<DATA_ROOT>/tiny-imagenet-200/val/val_annotations.txt
```

或：

```text
<DATA_ROOT>/tiny-imagenet-200/val/images/<class>/*.JPEG
```

在 WSL 中训练 Tiny-ImageNet 时，建议把数据放在 Linux 文件系统内；从 `/mnt/c` 或 `/mnt/d` 扫描大量小文件会明显变慢。
当前本地配置使用 `/mnt/d/Code/Relupruner/Relupruner/data`；上传或换机器后只需修改 `configs/tinyimagenet_resnet18.env` 中的 `DATA_ROOT`。

## 统一训练入口

最常用的完整训练命令：

```bash
bash cifar10.sh
bash cifar100.sh
bash tinyimagenet.sh
```

三个快捷脚本都调用 `run_experiment.sh`。单独运行某个阶段：

```bash
bash cifar10.sh teacher
bash cifar10.sh prune
bash cifar10.sh finetune
bash cifar10.sh evaluate
```

查看模型的可剪 ReLU 总量而不训练：

```bash
bash cifar10.sh info --model resnet34
```

临时覆盖模型、绝对预算、数据路径或批大小：

```bash
bash cifar10.sh all \
  --model resnet34 \
  --target-relus 49152 \
  --batch-size 64
```

可用覆盖项为 `--model`、`--target-relus`、`--keep-ratio`、`--data-root`、`--batch-size`、`--workers` 和 `--force`。

## 多模型、多预算训练

矩阵配置已经写好。先预览将要执行的命令：

```bash
bash run_matrix.sh configs/cifar10_matrix.env --dry-run
```

确认后启动完整矩阵：

```bash
bash run_matrix.sh configs/cifar10_matrix.env
bash run_matrix.sh configs/cifar100_matrix.env
bash run_matrix.sh configs/tinyimagenet_matrix.env
```

矩阵配置中的 `MODELS`、`BUDGETS` 和 `MODEL_BATCH_SIZES` 可以直接修改。跨模型比较建议使用 `BUDGET_TYPE=target_relus`，确保不同网络保留相同的绝对 ReLU 数量；`keep_ratio` 只适合比较各模型自身的保留比例。

teacher checkpoint 按 `(数据集, 模型)` 保存，与 ReLU 预算无关。因此：

- 同一模型更换预算时自动复用 teacher；
- 更换模型时只为新模型训练一次 teacher；
- `SKIP_COMPLETED_STAGES=1` 时自动复用已经完成的阶段；
- `--force` 会覆盖已有结果，矩阵训练通常不应使用。

更完整的字段说明见 [configs/README.md](configs/README.md)。

## 快速检查

以下配置只处理极少数 batch，用于检查新机器、依赖或代码改动：

```bash
bash run_experiment.sh configs/smoke_cifar10.env all
bash run_experiment.sh configs/smoke_cifar100.env all
bash run_experiment.sh configs/smoke_tinyimagenet.env all \
  --data-root /path/to/tiny-imagenet-parent
```

不要使用 smoke 输出报告模型准确率。

## 目录结构

```text
relupruner/
  importance.py     # 一阶/二阶 Taylor 重要性与层权重
  pruning.py        # MaskedReLU、全局预算和渐进式调度
  models/           # ResNet 与 WRN-22-8
  losses.py         # KD 与 PRAM 损失
  engine.py         # 训练和评估循环
  checkpoints.py    # 新旧 checkpoint 兼容读取
scripts/
  train_teacher.py
  train_pruner.py
  finetune.py
  evaluate.py
configs/            # 单实验、矩阵和 smoke 配置
tests/              # 公式、预算、模型、数据和 checkpoint 测试
```

CIFAR 输入下，ResNet18、ResNet34 和 WRN-22-8 的可剪 ReLU 总量分别为 491,520、901,120 和 1,392,640。Tiny-ImageNet 输入下，ResNet18 和 ResNet34 分别为 1,966,080 和 3,604,480。

## 测试

```bash
python -m pytest -q
python -m ruff check .
```

## 引用

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

## 许可证

本仓库采用 MIT License，详见 [LICENSE](LICENSE)。

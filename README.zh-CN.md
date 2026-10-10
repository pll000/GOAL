# GOAL：全局自维持与局部继承

[English](README.md) · [论文](https://doi.org/10.1016/j.patcog.2024.110679) · [MIT 许可证](LICENSE)

论文 **Global self-sustaining and local inheritance for source-free unsupervised domain adaptation** 的 PyTorch 训练代码，发表于 Pattern Recognition 155 (2024), 110679。

作者：Lin Peng、Yuhang He、Shaokun Wang、Xiang Song、Songlin Dong、Xing Wei、Yihong Gong。

GOAL 只使用无标签目标域图片，将源域训练好的分类器适应到目标域。方法结合 Mean Teacher、全局一致性自维持（GCSS）和局部拓扑继承（LTI）。

当前版本提供 **Office-Home + ImageNet V2** 配置，不包含数据集图片、训练好的模型或已完成的实验记录。论文中的 Office-Home 平均准确率为 74.1%，这是论文参考结果，并非当前发布版本已经验证的运行结果。

## 环境

使用 Linux，或已配置 NVIDIA GPU 的 WSL2。批量调度脚本依赖 Unix 文件锁和进程接口，不支持原生 Windows。建议 Python 3.10；NVIDIA 驱动需兼容固定依赖中的 CUDA 11.7。所有命令均在仓库根目录执行：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 数据

自行准备 Office-Home 数据，目录包含 `Art/`、`Clipart/`、`Product/` 和 `RealWorld/`（也支持 `Real_World/`），每个域下面按类别存放图片。

```bash
export DATA_ROOT=/path/to/officeHome
python scripts/prepare_data.py \
  --data-root "$DATA_ROOT" --output data/officehome --strict-counts
```

四个域分别包含 2,427、4,365、4,439、4,357 张图片，共享 65 个类别。列表中的目标域标签仅用于评估，不参与适应损失或模型选择。复用外部源模型时，需要核对类别编号；可通过 `--class-map` 指定源模型使用的类别映射。

## 训练

在物理 GPU 0 上依次训练四个源模型：

```bash
PYTHON="$(command -v python)" bash scripts/train_source.sh "$DATA_ROOT" 0
```

源模型使用 ResNet-50 `IMAGENET1K_V2` 初始化、全部源域数据、20 个 epoch、batch size 64、标签平滑 0.1、随机种子 2021，并选用最终 epoch 权重。首次运行会下载 ImageNet 权重，源模型保存在 `runs/source/{A,C,P,R}/`。

运行全部 12 个目标域适应任务：

```bash
python scripts/run_officehome.py \
  --data-root "$DATA_ROOT" \
  --source-root runs/source \
  --output-root runs/target \
  --gpus 0
```

`--gpus 0,1,2` 可使用多张物理 GPU 分配独立任务；`--pairs A2C,C2A` 指定迁移方向；准备好源权重后可用 `--dry-run` 检查命令；`--resume` 从 epoch 边界恢复未完成的任务，已完成且配置匹配的任务自动跳过。

[默认配置](configs/officehome_goal_v2.json) 为 40 个目标 epoch、batch size 64、瓶颈层及分类器学习率 0.001、骨干学习率 0.0001、5 个邻居和 5 个重要特征维度。教师参数 EMA 与预测 EMA 的动量均为 0.9，分离项和拓扑项系数均为 1.0，分离项不衰减。目标适应固定使用最终模型，不按目标域准确率挑选权重。

## 评估与结果

```bash
python scripts/summarize.py --runs runs/target

python scripts/evaluate.py \
  --run runs/target/goal_v2/seed2021/A2C \
  --model student --device cuda:0
```

任务目录为 `runs/target/goal_v2/seed2021/<任务>/`，保存 `last.pt`、`run.json`、`metrics.jsonl`、`stdout.log` 和 `summary.json`。学生、教师结果分别汇总，所有迁移任务齐全后才计算完整基准平均值。`--tta flip` 与 `--tta ten_crop` 是额外的学生模型推理协议。

## 实现说明

复现实验时，需要记录当前配置的具体选择：

- 使用 ImageNet V2 初始化源模型。
- 邻居置信度默认采用 Shannon 熵；论文公式（4）写为交叉熵形式。
- 分离项默认 `sum`，对锚点取平均，但没有公式（5）中再除以 batch size 减一的归一化。
- 拓扑距离使用 L2 归一化后的学生特征，只计算高置信同类别有效边，并对有效无向边取平均。
- 节点可靠性权重不参与梯度计算。
- 熵阈值每个 epoch 更新一次，来源是记忆库中各样本最近一次存储的教师即时预测。

其他熵、拓扑模式见 `goal/losses.py`。互惠近邻重加权和额外的邻居交叉熵损失属于可选扩展，当前配置均未启用。工具代码也支持 Office-31、ImageCLEF，但本版本仅附带 Office-Home 配置和列表。模块说明及 BibTeX 引用见 [英文 README](README.md)。

## 许可证

本仓库代码使用 [MIT 许可证](LICENSE)。论文、数据集、依赖项及外部下载权重保留各自的许可条款。

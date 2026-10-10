# GOAL：全局自维持与局部继承

[English](README.md) · [论文](https://doi.org/10.1016/j.patcog.2024.110679) · [MIT 许可证](LICENSE)

论文 **Global self-sustaining and local inheritance for source-free unsupervised domain adaptation** 的 PyTorch 训练代码，发表于 Pattern Recognition 155 (2024), 110679。

作者：Lin Peng、Yuhang He、Shaokun Wang、Xiang Song、Songlin Dong、Xing Wei、Yihong Gong。

GOAL 只使用无标签目标域图片，将源域训练好的分类器适应到目标域。方法结合 Mean Teacher、全局一致性自维持（GCSS）和局部拓扑继承（LTI）。

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

目标适应固定使用最终模型，不按目标域准确率挑选权重。

## 评估与结果

```bash
python scripts/summarize.py --runs runs/target

python scripts/evaluate.py \
  --run runs/target/goal_v2/seed2021/A2C \
  --model student --device cuda:0
```

任务目录为 `runs/target/goal_v2/seed2021/<任务>/`，保存 `last.pt`、`run.json`、`metrics.jsonl`、`stdout.log` 和 `summary.json`。学生、教师结果分别汇总，所有迁移任务齐全后才计算完整基准平均值。`--tta flip` 与 `--tta ten_crop` 是额外的学生模型推理协议。

模块说明及 BibTeX 引用见 [英文 README](README.md)。

## 许可证

本仓库代码使用 [MIT 许可证](LICENSE)。论文、数据集、依赖项及外部下载权重保留各自的许可条款。

# Advanced Machine Learning: 从零训练语言模型

目标不是拼装模型 API，而是建立可解释、可复现、可测量的实验系统。任务背景由学习者提供：唐杰教授相关作业；原始作业页面尚未核实，不能将这里的路线当作官方课程要求。

面向有多年编程经验的工程师：不从 Python 语法讲起，从数据契约、张量形状、数值稳定性和实验验收讲起。允许 PyTorch 提供张量、自动微分、AdamW 和设备运行时；**BPE、线性层、RMSNorm、RoPE、Attention、SwiGLU、Transformer 和交叉熵自行实现**。不使用 Hugging Face tokenizer、现成 Transformer 层、预训练权重或生产代码中的 SDPA。

**Week 1 入口：[七天任务与逐行代码讲解](docs/week1/README.md)。** 已在本地 M4 Mac 上真实训练 98,323,200 参数的模型，完成 100 步 / 12,800 token 的教学预算及中断恢复。验证 loss 从 8.83 降到 3.51，但生成仍重复；这是工程闭环，不是充分预训练。[英文实验记录、真实曲线与失败样例](reports/week1/README.md)均已保存。权重保留在本地，未提交 Git。

训练结果、原始日志和 checkpoint 核验的 **[真实截图及完整执行命令](reports/week1/README.md#screenshot-evidence-and-executed-commands)** 已随报告提交。截图是在训练后读取记录与真实权重生成的，不伪装成训练时的终端画面。

## 目录与阶段

| 路径 | 用途 / 当前状态 |
| --- | --- |
| `aml/` | 第一阶段可运行代码：分词、数据、模型、训练、生成 |
| `configs/` | 小模型闭环、98M Mac 教学配置、98M 大预算配置 |
| `examples/` | 本仓库编写的微型语料，只用于调试 |
| `tests/` | 正确性、因果性、过拟合、断点恢复测试 |
| [docs/references.md](docs/references.md) | 公开资料调查与证据边界 |
| [docs/week1/](docs/week1/README.md) | Week 1 任务、六个核心模块逐行讲解、测试与配置说明 |
| [stages/01-basics/](stages/01-basics/README.md) | 当前学习入口：从零组件与 0.1B 训练 |
| [stages/02-systems/](stages/02-systems/README.md) | 后续：Triton、多卡训练与推理对照 |
| [stages/03-data-scaling/](stages/03-data-scaling/README.md) | 后续：原始 dump、清洗、规模法则 |
| [stages/04-alignment/](stages/04-alignment/README.md) | 后续：同基座 SFT / DPO / RLVR |
| [stages/05-agents/](stages/05-agents/README.md) | 后续：可验证环境、Harness、长程 Agent |
| [reports/](reports/README.md) | Week 1 实测记录，以及最终英文报告与现场 Demo 的要求 |
| `data/`, `runs/` | 本地数据和实验产物，忽略提交 |

## 先跑通，再逐个拆解

在仓库根目录运行。推荐 Python 3.12；不要直接在系统 Python 安装。

```bash
uv sync --locked --python 3.12
source .venv/bin/activate

python -m unittest discover -s tests -v
python -m aml.tokenizer --input examples/train.txt \
  --output data/smoke-tokenizer.json --vocab-size 512
python -m aml.data --train examples/train.txt --val examples/val.txt \
  --tokenizer data/smoke-tokenizer.json --output data/smoke
python -m aml.train --config configs/smoke.json --data data/smoke \
  --output runs/smoke --device cpu --threads 1
python -m aml.generate --checkpoint runs/smoke/latest.pt \
  --prompt "The robot" --temperature 0 --max-new-tokens 40 --device cpu
```

没有 `uv` 时用 `python3.12 -m venv .venv` 和 `.venv/bin/python -m pip install -e .`。数据目录与运行目录不可复用覆盖；重复实验换一个新目录。

**上面的 smoke 命令只训练 14 万参数的小模型，不是 0.1B 实验。** 真正的 98M Mac 教学训练见 [runbook](docs/week1/runbook.md)，使用 `configs/100m-mac.json`；`configs/100m.json` 则是尚未执行的 20,000 步大预算实验。当前实现是单设备 FP32、显式二次方 Attention、无 KV cache 的正确性基线；后续阶段再优化，避免先把模型接线错误隐藏在性能框架里。

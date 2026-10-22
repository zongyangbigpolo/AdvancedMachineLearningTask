# 配置与工程文件

## 模型与训练配置

配置：[smoke.json](../../configs/smoke.json)、[100m-mac.json](../../configs/100m-mac.json)、[100m-local.json](../../configs/100m-local.json)、[100m.json](../../configs/100m.json)。

| 字段 | 含义 | smoke / 100 步 / 1,500 步 / 20,000 步配置 |
| --- | --- | --- |
| `model` | 模型结构；恢复时须与快照一致 | 四个配置均包含 |
| `vocab_size` | embedding 与输出 logits 的词表槽位数 | 512 / 8192 / 8192 / 8192 |
| `context_length` | 每个训练窗口的预测位置数 | 32 / 128 / 128 / 512 |
| `d_model` | embedding、Attention 输出、残差流宽度 | 64 / 768 / 768 / 768 |
| `n_layers` | Transformer block 数 | 2 / 12 / 12 / 12 |
| `n_heads` | Attention 头数；每头宽度为 D/H | 4 / 12 / 12 / 12 |
| `d_ff` | SwiGLU gate/up 的中间宽度 | 192 / 2304 / 2304 / 2304 |
| `train` | 更新预算和优化过程 | 四个配置均包含 |
| `steps` | optimizer 更新总数 | 40 / 100 / 1500 / 20000 |
| `batch_size` | 一个 micro-batch 的窗口数 | 4 / 1 / 1 / 2 |
| `grad_accum` | 每 update 的 micro-batch 数 | 1 / 1 / 1 / 16 |
| `learning_rate` | warmup 后的峰值学习率（Learning Rate, LR） | 0.003 / 0.0003 / 0.0003 / 0.0003 |
| `min_lr_ratio` | 最后 LR 为峰值的十分之一 | 均为 0.1 |
| `warmup_steps` | 线性升 LR 的 update 数 | 4 / 10 / 50 / 200 |
| `weight_decay` | 矩阵参数的 AdamW 衰减 | 均为 0.1 |
| `grad_clip` | 全局梯度裁剪阈值 | 均为 1.0 |
| `eval_interval` | 评估与保存的 update 间隔 | 10 / 10 / 150 / 200 |
| `eval_batches` | 固定随机验证窗口的 batch 数 | 4 / 4 / 16 / 20 |
| `seed` | 初始化与独立采样器种子 | 均为 42 |

后三份配置的模型参数量相同，训练预算、上下文与有效 batch 不同。实验结果对应 100 步和 1,500 步配置；20,000 步配置尚未运行。

## pyproject.toml

源码：[pyproject.toml](../../pyproject.toml)。

| 行 | 含义 |
| --- | --- |
| 1 | 声明构建工具配置区。 |
| 2 | editable 安装至少需要 setuptools 68。 |
| 3 | 使用标准 setuptools 构建后端。 |
| 5 | 声明 Python 项目元数据。 |
| 6 | 分发包名，不是 Python import 路径；import 使用 `aml`。 |
| 7 | 当前版本 0.1.0。 |
| 8 | 项目用途描述。 |
| 9 | 支持 Python 3.11 及以后；实验环境为 Python 3.12。 |
| 10 | 唯一直接运行时依赖是 PyTorch；它本身可能安装传递依赖。 |
| 12 | setuptools 的包发现设置。 |
| 13 | Python 包只包含 `aml*`；数据、实验和文档单独存放。 |

`uv.lock` 固定依赖版本、平台选择和下载摘要；`uv sync --locked` 按该文件创建环境。

## 其余文件

`aml/__init__.py` 标记可导入包，导入时不启动训练。`examples/train.txt` 和 `examples/val.txt` 是调试文本，每个非空行代表一个文档；训练实验使用 TinyStories 子集。

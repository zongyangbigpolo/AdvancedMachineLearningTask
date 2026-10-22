# 配置与工程文件

## 三份模型配置逐项解释

源码：[smoke.json](../../configs/smoke.json)、[100m-mac.json](../../configs/100m-mac.json)、[100m.json](../../configs/100m.json)。JSON 的大括号 / 逗号只表示嵌套和分隔，字段才是运行语义。

| 字段 | 含义 | smoke / Mac / 大预算 |
| --- | --- | --- |
| `model` | 决定模型结构；恢复时不能擅改 | 三个配置均包含 |
| `vocab_size` | embedding 与输出 logits 的词表槽位数 | 512 / 8192 / 8192 |
| `context_length` | 每个训练窗口的预测位置数 | 32 / 128 / 512 |
| `d_model` | embedding、Attention 输出、残差流宽度 | 64 / 768 / 768 |
| `n_layers` | Transformer block 数 | 2 / 12 / 12 |
| `n_heads` | Attention 头数；每头宽度为 D/H | 4 / 12 / 12 |
| `d_ff` | SwiGLU gate/up 的中间宽度 | 192 / 2304 / 2304 |
| `train` | 决定更新预算和优化过程 | 三个配置均包含 |
| `steps` | optimizer 更新总数 | 40 / 100 / 20000 |
| `batch_size` | 一个 micro-batch 的窗口数 | 4 / 1 / 2 |
| `grad_accum` | 每 update 的 micro-batch 数 | 1 / 1 / 16 |
| `learning_rate` | warmup 后的峰值 LR | 0.003 / 0.0003 / 0.0003 |
| `min_lr_ratio` | 最后 LR 为峰值的十分之一 | 均为 0.1 |
| `warmup_steps` | 线性升 LR 的 update 数 | 4 / 10 / 200 |
| `weight_decay` | 矩阵参数的 AdamW 衰减 | 均为 0.1 |
| `grad_clip` | 全局梯度裁剪阈值 | 均为 1.0 |
| `eval_interval` | 评估与保存的 update 间隔 | 10 / 10 / 200 |
| `eval_batches` | 固定随机验证窗口的 batch 数 | 4 / 4 / 20 |
| `seed` | 初始化与独立采样器种子 | 均为 42 |

Mac 和大预算配置参数量相同，但实验预算、上下文与有效 batch 不同，不能把它们的 loss 或吞吐当作同条件比较。

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
| 9 | 支持 Python 3.11 及以后；本次实测 Python 3.12。 |
| 10 | 唯一直接运行时依赖是 PyTorch；它本身可能安装传递依赖。 |
| 12 | setuptools 的包发现设置。 |
| 13 | Python 包只包含 `aml*`；数据、实验和文档单独存放。 |

`uv.lock` 是 `uv lock` 生成的依赖版本、平台选择与下载摘要，不是手写算法。`uv sync --locked` 按这个锁文件安装，避免只看宽版本区间造成环境漂移。不要逐行手改几百行平台 wheel URL。

## 其余文件

`aml/__init__.py` 的第 1 行是包说明，使 `aml` 成为可导入包；没有导入时自动训练等副作用。`examples/train.txt` 和 `examples/val.txt` 是本仓库自编调试文本，不是正式语料；每个非空行代表一个文档。

| `.gitignore` 行 | 含义 |
| --- | --- |
| 1 `.venv/` | 不提交安装后的依赖和解释器环境。 |
| 2 `__pycache__/` | 不提交 Python 解释器缓存。 |
| 3 `*.py[cod]` | 不提交字节码、优化字节码等编译产物。 |
| 4 `*.egg-info/` | 不提交 editable 安装元数据。 |
| 5 `.pytest_cache/` | 即使有人用 pytest，也不提交它的缓存。 |
| 6 `data/` | 本地数据工作目录；可分发的小子集另存为阶段归档。 |
| 7 `runs/` | 本地训练工作目录；报告保存在 reports，模型另按文件大小打包或上传 Release。 |
| 8 `.DS_Store` | 忽略 macOS Finder 自动产生的目录元数据。 |

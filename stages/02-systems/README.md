# 第二阶段：注意力与生成性能

[总任务表](../../README.md) · [上一阶段](../01-basics/README.md) · [下一阶段](../03-data-scaling/README.md)

## 任务清单

- [x] 对比手写注意力与 PyTorch SDPA 的输出、梯度及耗时。
- [x] 手写逐层 KV Cache，检查分块预填充与逐 token 解码结果。
- [x] 在 Mac 上测量有无缓存的生成延迟、输出一致性及缓存大小。
- [ ] Triton Attention Kernel：当前 Mac 无兼容设备，未实现与实测。
- [ ] 多 GPU 训练与推理增益：当前只有一张 Apple GPU，未实测。

## 实现与名词

| 名词与英文全称 | 本阶段做法 |
| --- | --- |
| GPU — Graphics Processing Unit | 本机 Apple M4 图形处理器。 |
| MPS — Metal Performance Shaders | 本次计时使用的 PyTorch 后端。 |
| FP32 — 32-bit Floating Point | 所有对照都使用相同精度。 |
| SDPA — Scaled Dot-Product Attention | 缩放点积注意力；调用 PyTorch 实现作数值与性能对照。 |
| KV Cache — Key-Value Cache | 保存每层历史键和值；生成新 token 时只计算新位置。 |
| RoPE — Rotary Position Embedding | 缓存解码时必须沿用原位置偏移，不能每个新 token 都从位置 0 开始。 |
| Kernel，计算内核 | 在 GPU 上执行的计算程序；Triton 是编写此类程序的语言与工具名称，不是缩写。 |
| CUDA — Compute Unified Device Architecture | NVIDIA GPU 的计算平台；本机没有该设备。 |
| DDP — Distributed Data Parallel | 各卡保留模型副本、处理不同数据并同步梯度；尚未做硬件实验。 |

[experiment.py](experiment.py) 保留显式注意力作参考，另写 `CachedDecoder`。缓存逐层保留 K/V，追加时只对新 Q/K 加入正确位置编码。超过上下文就报错，不能简单丢弃旧缓存后声称与滑动窗口模型相同。

[测试](test_experiment.py)覆盖了多样本、分块预填充、逐步 logits、缓存字节数、上下文越界、批量变化，以及注意力反向梯度。

## 实验记录

在没有其他模型训练运行时计时。每轮先预热两次，计时前后等待 GPU 完成计算，避免把异步提交时间当成实际计算时间。表内是重复运行的中位数，完整最小 / 最大值见 [results.json](reports/results.json)。

**注意力前向 + 反向，重复 7 次，单位 ms**

| 序列长度 | 手写版 | PyTorch SDPA | 加速比 | 输出 / 梯度最大绝对误差 |
| --- | --- | --- | --- | --- |
| 64 | 1.0329 | 0.9403 | 1.098× | 0 / 0 |
| 128 | 1.1378 | 1.1120 | 1.023× | 0 / 0 |
| 256 | 1.1286 | 1.1071 | 1.019× | 0 / 0 |

**实际贪心生成，32-token 输入，重复 5 次，单位 ms**

| 新生成 token | 每次重算前文 | 使用缓存 | 加速比 | 输出相同 | 缓存 bytes |
| --- | --- | --- | --- | --- | --- |
| 16 | 107.469 | 98.358 | 1.093× | 是 | 48,128 |
| 32 | 209.511 | 190.206 | 1.101× | 是 | 64,512 |
| 64 | 431.405 | 390.203 | 1.106× | 是 | 97,280 |

生成对照使用 2 层、宽度 64 的固定随机权重模型；测的是实现成本，不是文本质量。注意力测试的形状为 batch=1、4 heads、head_dim=16。

## 复盘

缓存消除了历史位置的重复投影和前馈计算，但这里网络小、序列短，运行调度占比高，收益只有约 10%。注意力两种实现的计时区间也有重叠，不能从这些数字推断大模型或长序列同样有这个加速比，更不能将 PyTorch SDPA 结果写成自研 Triton 成绩。

## 实际命令

第一阶段环境安装后，从仓库根目录运行：

```bash
cd stages/02-systems
../01-basics/.venv/bin/python -m unittest -v
../01-basics/.venv/bin/python experiment.py --device mps
../01-basics/.venv/bin/python ../01-basics/scripts/render_reports.py --stage 02-systems
```

重跑会重写本阶段的性能报告；不同负载下结果可能变化。核心代码和完整测量都保存在本阶段，不依赖别人仓库里的运行结果。

## 实验截图

浏览器截图对应 [原始测量页面](reports/index.html)，不是运行中的终端。

**注意力数值和前后向性能**

![注意力对照](reports/attention.png)

**相同输出下的缓存收益**

![生成缓存对照](reports/cache.png)

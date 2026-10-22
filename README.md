# 高级机器学习实验作业

研究从原始文本到模型训练、后训练和工具调用的完整流程。实验包括约一亿参数语言模型训练、注意力与解码优化、语料清洗与规模拟合、三种后训练方法对照，以及多步智能体任务。

| 阶段 | 任务 | 实验结果 |
| --- | --- | --- |
| [01 基础](stages/01-basics/README.md) | 手写分词器和 Transformer，训练约一亿参数模型 | 98M 模型训练 1,500 步，验证交叉熵从 8.9119 降至 2.7503 |
| [02 系统](stages/02-systems/README.md) | 注意力与生成性能 | 缓存解码加速约 1.10 倍；Triton 内核与多卡实验未完成 |
| [03 数据与规模](stages/03-data-scaling/README.md) | 原始数据清洗、规模实验、拟合与外推 | 原始 XML 清洗、15 次训练、最大模型留出误差 0.86% |
| [04 对齐](stages/04-alignment/README.md) | 同基座监督微调、偏好优化、可验证奖励强化学习对照 | 共 9 次训练；数字任务平均正确率 90% / 90% / 85% |
| [05 智能体](stages/05-agents/README.md) | 可验证环境、多步策略训练与自评循环 | 16 目标任务完成 12/12，平均 68.08 次工具动作；自评增加调用成本并改善扰动条件下的成功率 |

## 实验环境与复现

硬件为 Apple M4、16 GiB 统一内存。基础训练与性能实验使用 MPS（Metal Performance Shaders）后端，其余小模型实验使用 CPU（Central Processing Unit）单线程。依赖环境：

```bash
cd stages/01-basics
uv sync --locked --python 3.12
```

[训练快照下载](https://github.com/zongyangbigpolo/AdvancedMachineLearningTask/releases)（每份约 1.18 GB，包含优化器和随机状态）。

## 报告与演示

[英文报告](stages/05-agents/report/paper.pdf) · [策略运行演示](stages/05-agents/demo.py) · [参考资料](stages/01-basics/docs/references.md)

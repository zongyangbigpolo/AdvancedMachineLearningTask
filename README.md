# 高级机器学习实验作业

按五个阶段保存实现、实验记录和截图。每个阶段的 README 是任务清单与实验报告，代码和产物都放在对应目录。

| 阶段 | 任务 | 已提交记录 |
| --- | --- | --- |
| [01 基础](stages/01-basics/README.md) | 手写分词器和 Transformer，训练约一亿参数模型 | 98M 模型 1,500 步、分词对照、生成、五张截图、数据包及权重下载 |
| [02 系统](stages/02-systems/README.md) | 注意力内核、多卡、训练与推理性能 | Mac 注意力与 KV 缓存实测；Triton、多卡待设备 |
| [03 数据与规模](stages/03-data-scaling/README.md) | 原始数据清洗、规模实验、拟合与外推 | 待提交 |
| [04 对齐](stages/04-alignment/README.md) | 同一基座的 SFT、DPO、RLVR 对照 | 待提交 |
| [05 智能体](stages/05-agents/README.md) | 可验证环境、运行框架、长程训练与自评循环 | 待提交 |

本次使用本机 Apple M4，不租 GPU。Triton 和多 GPU 实测保留为待补项，其余项目以各阶段实际记录为准。每阶段整理完单独提交并 push 到 `main`。

运行前先进入对应阶段目录。第一阶段安装依赖，后续阶段复用该 Python 环境：

```bash
cd stages/01-basics
uv sync --locked --python 3.12
```

小数据包和小模型随阶段保存；超过 Git 文件限制的大模型放同仓库 [Releases](https://github.com/zongyangbigpolo/AdvancedMachineLearningTask/releases)。下载命令与摘要见对应阶段 README。

最终交付为 NeurIPS（Neural Information Processing Systems）格式英文报告和现场 Demo（Demonstration，演示）。[公开参考与对照记录](stages/01-basics/docs/references.md)区分了第三方结果和本仓库实际测量。

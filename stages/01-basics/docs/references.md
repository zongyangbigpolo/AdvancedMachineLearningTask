# 参考资料与实现对照

## 项目和课程资料

| 资料 | 参考内容 |
| --- | --- |
| [Stanford CS336](https://stanford-cs336.github.io/spring2025/) | 分词器、模型、训练、系统优化、规模与后训练的课程组织 |
| [CS336 Basics](https://github.com/stanford-cs336/assignment1-basics) | 因果性、模型组件与训练流程的测试约束 |
| [CS336 Systems](https://github.com/stanford-cs336/assignment2-systems) | 注意力性能、分布式训练与推理的实验协议 |
| [CS336 Scaling](https://github.com/stanford-cs336/assignment3-scaling) | 固定预算的训练规模实验 |
| [CS336 Alignment](https://github.com/stanford-cs336/assignment5-alignment) | 监督、偏好与奖励信号的实现与评估 |
| [jackmcgradylee/from-token-to-agent](https://github.com/jackmcgradylee/from-token-to-agent/tree/ca6f58686f9a42bcf45be313c41595a0934e914a) | 字符 / 单词 / BPE 对照、架构消融、规模 pilot 与最大模型留出检验 |
| [karpathy/minbpe](https://github.com/karpathy/minbpe) | 字节 BPE（Byte Pair Encoding）的教学组织 |
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | 简洁的语言模型训练和生成接口 |
| [karpathy/nanochat](https://github.com/karpathy/nanochat) | 分词、预训练、后训练、评估与推理的连接 |
| [ShallowU/cs336-assignment](https://github.com/ShallowU/cs336-assignment) | CS336 A1、A2、A5 的公开实现记录 |

`from-token-to-agent` 的参考版本为
`ca6f58686f9a42bcf45be313c41595a0934e914a`。第三阶段采用其中的实验组织方式：
以较小模型拟合规模关系，将最大模型留作预测检验。

## 方法依据

| 方法 | 来源 |
| --- | --- |
| 缩放点积注意力数值对照 | [PyTorch SDPA 文档](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) |
| 幂律规模拟合 | Kaplan et al., [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), 2020 |
| 偏好优化 | Rafailov et al., [Direct Preference Optimization](https://arxiv.org/abs/2305.18290), 2023 |
| 策略梯度 | Williams, [Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://doi.org/10.1007/BF00992696), 1992 |
| 故事语料 | Eldan and Li, [TinyStories](https://arxiv.org/abs/2305.07759), 2023 |

## 实现取舍

| 部分 | 采用的实现 | 选择原因 |
| --- | --- | --- |
| 分词 | 字节起点、确定性 BPE、文档结束标记 | 可逆、便于手算与测试 |
| 模型 | 显式注意力、RoPE、RMSNorm、SwiGLU、共享嵌入 | 先建立可检查基线，再替换内部计算 |
| 数据 | 文件映射、小端 uint32、目标右移一位 | 降低存储开销并保留清晰输入契约 |
| 性能 | 同输入输出、同步计时、预热与重复运行 | 区分数值变化、首次编译与稳定耗时 |
| 规模 | 同分词器、同 token 预算、三个种子 | 保持模型大小之外的主要条件一致 |
| 后训练 | 同基座、同题集、同候选范围 | 比较学习信号及资源成本 |
| 智能体 | 受控环境、轨迹监督、终局奖励、复查对照 | 逐步检查行为、反馈和长程失败 |

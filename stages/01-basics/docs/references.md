# 公开实现调查

调查日期：2026-10-21。通过搜索引擎与 GitHub 仓库检索，并阅读下列官方页面、项目说明或元数据；未复跑第三方训练，也没有审查这些项目的全部实现。

## 结论与证据边界

**有人公开实现了高度相似的组件与训练流程，但尚不能确认有人完整完成了这里列出的“唐杰教授作业”。** 本次检索“唐杰 + 大作业 / RLVR / self-judge / 0.1B”等组合未找到可核实的原始要求。搜索无结果不等于不存在，不能据此断言“无人完成”，也不能确认作业归属、课程年份或官方验收规则。

CS336 官方教学路线与前四类要求高度重合，但这是相关参考，不是作业来源证明；尤其长程 Agent、self-judge、指定 NeurIPS 报告不能从这个相似性推出。

| 资料 | 查证到的内容 | 不能据此声称 |
| --- | --- | --- |
| [Stanford CS336 Spring 2025 官方课程页](https://stanford-cs336.github.io/spring2025/) | A1 从零 tokenizer / Transformer / 训练；A2 Triton FlashAttention 与分布式；A3 scaling；A4 原始 Common Crawl；A5 SFT / RL，DPO 为补充部分 | 是唐杰教授这份作业，或包含全部 Agent 要求 |
| [CS336 Assignment 1](https://github.com/stanford-cs336/assignment1-basics) | 官方作业入口、测试和任务要求 | 官方骨架本身就是完成解答 |
| [Assignment 1 排行榜](https://github.com/stanford-cs336/assignment1-basics-leaderboard) | 存在公开 loss / 实验链接；读取时页面已更新为 Spring 2026，验证状态列并非全部标记已复现 | 所有结果都经过我们或课程方复现；2025/2026 的硬件、时间预算可以混用 |
| [ShallowU/cs336-assignment](https://github.com/ShallowU/cs336-assignment) | README 自述实现 A1、A2、A5，附 Colab / Kaggle / 博客；作者明确提到参考与 AI 辅助 | 完成 A3、A4、DPO 全套、长程 Agent；或独立完成的官方认证 |
| [karpathy/minbpe](https://github.com/karpathy/minbpe) | MIT，byte-level BPE 教学项目 | 覆盖模型训练或整份作业 |
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | MIT，GPT 训练 / 微调参考项目 | tokenizer、Triton、RLVR、Agent 全部从零 |
| [karpathy/nanochat](https://github.com/karpathy/nanochat) | README 展示 tokenizer、预训练、微调、评估、推理闭环及公开实验记录 | 符合本作业全部约束，或直接复用即等于自己完成 |
| [jackmcgradylee/from-token-to-agent](https://github.com/jackmcgradylee/from-token-to-agent/tree/ca6f58686f9a42bcf45be313c41595a0934e914a) | 用户补充后阅读了其课程来源说明、W1 / HW1 文档和第一阶段核心源码；作者自述基于唐杰 2026-09-17 课程，附 PPT 拼图；包含 tokenizers 对照、架构消融、scaling pilot 等记录 | 个人整理即官方验收规则；README 的完成标记即全部 0.1B 正式实验完成，或参考源码没有错误 |

## 用户补充项目后的对照检查

本仓库最初版本并未参考该项目源码。以上补充对照固定在 `ca6f58686f9a42bcf45be313c41595a0934e914a`；没有运行或复制第三方训练代码。其 [HW1 README](https://github.com/jackmcgradylee/from-token-to-agent/blob/ca6f58686f9a42bcf45be313c41595a0934e914a/assignments/hw1-foundations/README.md) 标为 complete，但 [HW1 report](https://github.com/jackmcgradylee/from-token-to-agent/blob/ca6f58686f9a42bcf45be313c41595a0934e914a/assignments/hw1-foundations/report.md) 仍写 smoke only / full 0.1B pending，因此不能据此认定正式预训练已经完成。

对本仓库 `24c7ccd` 的结论：

| 分类 | 结论 |
| --- | --- |
| 核心数学 | 未发现 BPE merge 次序、target shift、因果遮罩、RoPE 配对、SwiGLU、共享输出权重或 CE 的确定性接线错误；与数值 oracle 和独立性质测试相符，不代表完整形式化证明 |
| 已复现的恢复缺陷 | `aml/train.py:172–174` 无条件解析每行 JSON；模拟进程中断留下半行日志时，即使 checkpoint 有效，resume 仍报 `JSONDecodeError` |
| 已确认的溯源缺陷 | `aml/train.py:138–140,160–169` 恢复只核对配置、数据、设备；源码指纹只在新实验写入，恢复时不检查或追加。构造与当前源码不一致的已记录指纹后，resume 仍完成并保留错误归因 |
| 实验覆盖缺口 | 本仓库按阶段组织，不与对方的 W1 等周次对应；没有 Char / Word / BPE 定量对照，也没有 RMSNorm / LayerNorm、MHA / GQA 消融。应按需补实验，不把第一阶段教学训练等同于对方所有任务完成 |
| 训练规模局限 | 98M 参数的 100 步只证明教学闭环，不能代替充分预训练；固定验证窗口仅 512 个预测位置，不能当作完整能力评估 |

上述两个恢复缺陷在本次检查中尚未修复。截图核验对已有原始实验额外检查了源码指纹，并不等于训练器的通用 resume 已获得该保护。原始 100 步训练的权重、日志和已保存源码指纹通过了这次独立核验，没有发现这两类问题在原始实验中发生。

## 如何参考而不失去学习价值

先读概念与测试约束，自己写组件，再用第三方项目解释取舍；本仓库没有复制上述项目源码。阅读或复用第三方实现必须遵循其许可证并注明来源。若用于正式课程提交，应先确认所在课程的 AI / 合作规范；CS336 官方明确限制直接用 AI 生成作业解答，因此本仓库应视作独立学习实验，不能宣称是可直接提交的官方课程答案。

本仓库的实现顺序是“正确性基线 → 小模型闭环 → 0.1B 实验 → 性能 → 数据与规模 → 对齐 → Agent”。不同 tokenizer 下的 token-level loss 不宜直接比较；排行榜数字也不能当成本项目已经取得的结果。

## 2026-10-22 的阶段实验对照

再次读取 `from-token-to-agent` 的英文 README、完整目录树，以及 RLVR / Harness 入口；固定版本仍为 `ca6f58686f9a42bcf45be313c41595a0934e914a`。该版本后两者仍是说明性入口，README 也将后训练和最终项目列为未开始，因此没有把它当成已完成全套作业的答案。

这次参考它的字符 / 单词 / BPE 序列对照、固定数据量的规模实验和留出最大模型的做法；本仓库自行实现并实跑，不复制其训练代码或实验数字。第一阶段新结果为 98M 参数、1,500 步、192,000 token，验证 CE 从 8.9119 到 2.7503；原 100 步证据保持原样。

后续方法的资料入口：

| 实验 | 方法依据 |
| --- | --- |
| 手写缓存与注意力核对 | [CS336 Systems](https://github.com/stanford-cs336/assignment2-systems)、[PyTorch SDPA 文档](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) |
| 固定 token 预算的模型规模拟合 | [CS336 Scaling](https://github.com/stanford-cs336/assignment3-scaling)、[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)；不声称复现完整 Chinchilla 数据 / 计算联合拟合 |
| 偏好与可验证奖励训练 | [CS336 Alignment](https://github.com/stanford-cs336/assignment5-alignment)、[Direct Preference Optimization](https://arxiv.org/abs/2305.18290) |
| 策略梯度 | Williams, 1992, [Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://doi.org/10.1007/BF00992696)；用外部结果奖励，不把监督答案损失改名为 RL |

实验规模、候选输出约束、计时范围、数据划分和未覆盖硬件均在各阶段 README 及机器可读记录中写明。

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
| 实验覆盖缺口 | 这里的自定 Week 1 不等于对方的 W1；没有 Char / Word / BPE 定量对照，也没有 RMSNorm / LayerNorm、MHA / GQA 消融。应按需补实验，不把不同周次混称为完成 |
| 训练规模局限 | 98M 参数的 100 步只证明教学闭环，不能代替充分预训练；固定验证窗口仅 512 个预测位置，不能当作完整能力评估 |

上述两个恢复缺陷在本次检查中尚未修复。截图核验对已有原始实验额外检查了源码指纹，并不等于训练器的通用 resume 已获得该保护。原始 100 步训练的权重、日志和已保存源码指纹通过了这次独立核验，没有发现这两类问题在原始实验中发生。

## 如何参考而不失去学习价值

先读概念与测试约束，自己写组件，再用第三方项目解释取舍；本仓库没有复制上述项目源码。阅读或复用第三方实现必须遵循其许可证并注明来源。若用于正式课程提交，应先确认所在课程的 AI / 合作规范；CS336 官方明确限制直接用 AI 生成作业解答，因此本仓库应视作独立学习实验，不能宣称是可直接提交的官方课程答案。

本仓库的实现顺序是“正确性基线 → 小模型闭环 → 0.1B 实验 → 性能 → 数据与规模 → 对齐 → Agent”。不同 tokenizer 下的 token-level loss 不宜直接比较；排行榜数字也不能当成本项目已经取得的结果。

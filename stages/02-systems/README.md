# 第二阶段：性能优化（尚未实现）

先冻结第一阶段的正确性基线，再实现 Triton Attention，并比较前向 / 反向误差、延迟和显存。多卡训练和推理必须分别设计实验，记录硬件、通信互联、batch、precision、并行策略、warmup 与重复次数。

训练报告 tokens/s 和 scaling efficiency；推理分别报告首 token 延迟、逐 token 延迟与吞吐。保持任务与计算预算可比，不能把增大 batch 的吞吐直接称为单请求加速。需要 Linux + NVIDIA GPU 环境；本阶段不由 Mac smoke 代替。

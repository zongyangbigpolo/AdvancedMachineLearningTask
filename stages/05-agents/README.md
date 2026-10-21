# 第五阶段：长程 Agent（尚未实现）

先定义可重放、可验证的任务环境，再定义 Harness 的 observation / action / reward / termination 契约。轨迹应记录工具调用、环境状态、预算与验证结果；对执行设置隔离和权限边界。

先做短程任务和确定性基线，再增加任务链长度。self-judge 作为可选对照，与外部 verifier 分开：模型说“完成”不是成功判据，必须测量 verifier 下的成功率、循环次数与资源成本。

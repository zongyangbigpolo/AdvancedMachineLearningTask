# 第四阶段：同基座对齐对照（尚未实现）

冻结同一预训练 checkpoint、tokenizer 和评估协议，建立 SFT、DPO、RLVR 分支。记录数据、训练 token / 算力预算、随机种子与实际使用的起点；明确 DPO / RLVR 是否从共同 SFT checkpoint 出发，不混淆基座差异和算法差异。

RLVR 奖励必须来自可验证任务执行结果，并防止 reward hacking。结果应包含未参与训练的任务成功率、误差分析和成本，不仅是训练 reward。

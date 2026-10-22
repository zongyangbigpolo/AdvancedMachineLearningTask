# 第五阶段：可验证环境、长程智能体与自评循环

[总任务表](../../README.md) · [上一阶段](../04-alignment/README.md)

## 任务清单

- [x] 写可重置环境、三个工具、明确成功条件与失败条件。
- [x] 写 Harness，限制工具、步数和整数范围，保存完整任务轨迹。
- [x] 用成功轨迹训练神经网络策略，再用环境奖励更新。
- [x] 测试 1、4、8、16 个连续目标，分别比较训练前后与 self-judge。
- [x] 保存策略权重、示范数据、任务轨迹、截图与运行演示。
- [x] 整理五阶段英文报告，使用官方 NeurIPS 2026 模板。

## 任务环境

选择整数账本任务，使状态变化、动作依赖和终止条件都能逐步检查：

```text
当前值：1，目标值：6
允许的工具：
  L：减 1
  R：加 1
  S：提交当前目标

达到 6 并提交后，才会出现下一个目标。
全部目标依次提交且最终数值正确，任务才算成功。
```

模型每次看到当前值和目标，从三个工具中选一个。错误提交、超出 0～9 或耗尽动作预算都会终止任务。环境限定在内存账本中。

| 名词与英文全称 | 实现中的含义 |
| --- | --- |
| Agent，智能体 | 持续观察、调用工具、接收反馈的神经网络策略。 |
| Harness，执行框架 | 校验并执行模型请求，记录每步输入、草稿、实际动作和环境反馈。 |
| Observation / Action，观察 / 动作 | 环境给出的数值状态 / 模型选择的 L、R、S。 |
| Trajectory，任务轨迹 | 一次任务从开始到终止的完整动作与反馈。 |
| Verifier，验证器 | 同时检查所有最终数值、提交顺序和成功终止状态。 |
| SFT — Supervised Fine-Tuning | 从成功轨迹学习下一步动作，以及草稿复查动作。 |
| RL — Reinforcement Learning | 用执行后的终局奖励加权动作对数概率，更新策略。 |
| Self-judge，自我评估 | 同一个模型再读一遍“当前状态 + 自己的草稿动作”，给出复查动作。 |
| CPU — Central Processing Unit | 本次训练和推理都用 CPU 单线程。 |
| Demo — Demonstration | 重新加载保存的权重，现场执行任务并调用外部验证器。 |

环境、神经策略与执行框架都在 [experiment.py](experiment.py)。规则函数生成训练示范，推理阶段由模型选择动作；[测试](test_experiment.py)检查两条路径的隔离。

## 训练记录

| 项目 | 设置 |
| --- | --- |
| 起点 | 第四阶段 SFT、种子 42 的 289,248 参数模型 |
| 示范 | 从 1、2、4、8 目标的已验证轨迹得到 10,556 条普通 / 复查训练记录 |
| 轨迹 SFT | 300 次更新，每批 32 条 |
| 环境奖励训练 | 24 个采样 episode，逐渐增加到 3 个目标 |
| 奖励 | 终局外部验证成功为 1，否则 0；基线 0.5，熵系数 0.001 |
| 测试 | 每个目标长度 12 个新种子任务，最多 128 次工具动作 |

轨迹监督后，策略已在全部测试任务上成功；随后 24 个奖励训练 episode 的奖励均为 1，测试成功率保持不变。

复查提示属于监督训练样本。测试使用新的目标序列，数字状态和工具范围与训练一致。

## 实测结果

表内是完整任务成功数，每格共 12 次：

| 策略 / 条件 | 1 个目标 | 4 个目标 | 8 个目标 | 16 个目标 |
| --- | --- | --- | --- | --- |
| 智能体训练前 | 0 | 0 | 0 | 0 |
| 轨迹 SFT | 12 | 12 | 12 | 12 |
| 轨迹 SFT + 环境奖励训练 | 12 | 12 | 12 | 12 |
| 再加 self-judge | 12 | 12 | 12 | 12 |
| 单独压力测试：10% 草稿扰动、不复查 | 9 | 2 | 2 | 0 |
| 同一扰动设置、增加复查 | 12 | 12 | 12 | 12 |

16 目标任务平均 **68.08 次工具动作**；加复查后动作数没变，模型调用从 68.08 次增至 **136.17 次**。

普通任务全部成功，自评没有继续提高成功率，但增加了计算。10% 草稿随机扰动下，自评提高了成功率；该压力测试与普通任务分开统计。

## 任务轨迹

保存的 8 目标评估任务用 32 次动作完成：

```text
目标：     [6, 9, 5, 6, 2, 1, 6, 1]
最终账本： [6, 9, 5, 6, 2, 1, 6, 1]
提交顺序： 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7
外部验证： true
```

各策略及扰动条件下的任务轨迹见 [traces.json](reports/traces.json)。

## 复盘

有限状态控制器在新目标序列上完成了长动作链。轨迹监督贡献了主要收益，奖励训练后的成功率保持不变。复查提示的额外计算在扰动条件下有效，但也可能只是重复应用状态到动作的映射；复杂规划和新工具迁移仍需额外任务评估。

## 复现命令与模型

第四阶段模型包解压后：

```bash
cd stages/05-agents
mkdir -p runs
../01-basics/.venv/bin/python -m unittest -v
../01-basics/.venv/bin/python experiment.py --sft-steps 300 --rl-episodes 24 > runs/agent.stdout.json
../01-basics/.venv/bin/python demo.py --self-judge > reports/demo.jsonl
../01-basics/.venv/bin/python ../01-basics/scripts/render_reports.py --stage 05-agents
```

[agent-artifacts.zip](artifacts/agent-artifacts.zip)包含示范、分词器、SFT 权重、最终策略权重与任务轨迹。加载已有策略：

```bash
unzip -n artifacts/agent-artifacts.zip
../01-basics/.venv/bin/python demo.py --self-judge
```

演示执行 **36 步、72 次模型调用**，最终状态通过验证器检查。初始账本与前述 32 步评估任务不同，逐步输出见 [demo.jsonl](reports/demo.jsonl)。

重训可加 `--output runs/agent-repeat`，不会覆盖保存的模型。

## 英文报告

[英文 PDF](report/paper.pdf) · [LaTeX 源码](report/paper.tex) · [官方模板来源](report/TEMPLATE_SOURCE.txt)

采用 NeurIPS（Neural Information Processing Systems）2026 preprint 格式。使用 Tectonic 编译：

```bash
cd report
tectonic paper.tex
```

## 实验截图

**训练预算与奖励记录**

![智能体训练记录](reports/training.png)

**不同任务长度、复查与调用成本**

![长程任务和自评对照](reports/evaluation.png)

**轨迹首尾动作与终态检查**

![神经策略任务轨迹](reports/trajectory.png)
![重新加载权重的 Demo 验证](reports/verification.png)

# 实验截图、执行命令与检查脚本

查看[报告中的三张截图](../../reports/README.md#artifacts-and-screenshots)：

| 截图 | 展示什么 |
| --- | --- |
| [training-results.png](../../reports/evidence/training-results.png) | 损失曲线、参数规模与重复生成样例 |
| [checkpoint-verification.png](../../reports/evidence/checkpoint-verification.png) | checkpoint 的 SHA、参数数量、Adam update 和有限值检查 |
| [training-log.png](../../reports/evidence/training-log.png) | stdout 中的 step 0、5、6、100，对应起始、中断、恢复与结束 |

截图展示实验记录页面。完整命令见 [commands.txt](../../reports/evidence/commands.txt)，报告目录同时保存两段 stdout 和机器可读检查结果。

## 重做核验和截图

脚本读取 100 步实验的 checkpoint、`manifest.json`、metrics 和两段 stdout。权重可从 [stage1-100-updates](https://github.com/zongyangbigpolo/AdvancedMachineLearningTask/releases/tag/stage1-100-updates) 下载；每次新训练应使用独立输出目录与报告。

先从仓库根目录进入第一阶段：`cd stages/01-basics`。

```bash
.venv/bin/python -m scripts.evidence \
  > reports/checkpoint-verification.log
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 \
  --directory reports
```

打开 `http://127.0.0.1:8765/evidence/index.html`，分别截取 `#results`、`#audit` 和 `#logs`。浏览器 viewport 为 1280×1100，按元素保存 PNG；截图后停止预览服务器。[index.html](../../reports/evidence/index.html) 是对应页面。

## 核验脚本逐行说明

源码：[scripts/evidence.py](../../scripts/evidence.py)。脚本检查 100 步实验的训练快照与日志，并渲染报告页面。

| 源码行 | 含义 |
| --- | --- |
| 1 | 模块说明：读取 checkpoint 与日志，生成实验报告页。 |
| 3 | 解析本地实验和报告路径。 |
| 4 | 转义页面中的日志文本，避免把文本当成 HTML。 |
| 5 | 解析 checkpoint 旁的 JSON 和训练 JSONL。 |
| 6 | 将训练日志复制到报告目录。 |
| 7 | Path 管理文件。 |
| 9 | 读取张量状态并检查有限值。 |
| 11 | 流式计算大文件的 SHA。 |
| 14 | 开始复现命令字符串，首条命令创建日志目录。 |
| 15 | 原始 TinyStories 小集下载命令与 stdout 路径。 |
| 16 | 原始手写 BPE 训练命令。 |
| 17 | 原始 train / val token 准备命令。 |
| 18 | 原始 MPS 训练命令，在总计划第 5 步停止。 |
| 19 | 原始恢复命令，仍使用相同配置与数据。 |
| 20 | greedy 生成命令及输出路径。 |
| 22 | 检查与页面生成命令的分组标题。 |
| 23 | 执行快照检查与报告渲染。 |
| 24 | 结束多行命令文本。 |
| 27 | 定义脚本入口。 |
| 28 | 创建 CLI 解析器。 |
| 29 | 默认读取原始实验目录。 |
| 30 | 默认读取当前第一阶段下的 `reports/`。 |
| 31 | 解析参数。 |
| 32 | 核验在 CPU 单线程运行，避免不必要的资源占用。 |
| 33 | 读取原实验 summary。 |
| 34 | 读取运行 manifest。 |
| 35 | 设置权重路径。 |
| 36 | 将 checkpoint 的模型与 Adam 状态加载到 CPU。 |
| 37 | 配置与完成 step 必须和报告一致。 |
| 38 | 不一致就报错。 |
| 39 | 计算 checkpoint 文件的 SHA。 |
| 40 | 与 summary 记录的 checkpoint 指纹比较。 |
| 41 | 文件摘要不一致时报错。 |
| 42 | 累加权重张量元素数量。 |
| 43 | 核对参数总数。 |
| 44 | 数量不一致报错。 |
| 45 | 逐张量检查全部权重是否为有限数。 |
| 46 | 检测到 NaN / Inf 就报错。 |
| 47 | 读取每个参数的 Adam step，去重排序。 |
| 48 | 核对各参数 Adam 状态的更新次数。 |
| 49 | 优化器预算对不上就报错。 |
| 50 | 遍历训练开始时记录的每个源码指纹。 |
| 51 | 计算当前源码的 SHA 并比较。 |
| 52 | 源码摘要不一致时报错。 |
| 53 | 设置报告子目录。 |
| 54 | 创建目录，不删除已有截图。 |
| 55 | 原始前五步 stdout 路径。 |
| 56 | 原始恢复段 stdout 路径。 |
| 57 | 将两段原始 stdout 按顺序解析成记录。 |
| 58 | 将 stdout 记录与逐步 metrics 比较。 |
| 59 | 原始记录不一致时显式失败。 |
| 60 | step 必须完整覆盖 0 到 100，不能缺失或重复。 |
| 61 | 不满足就报错。 |
| 62 | 遍历两份原始日志。 |
| 63 | 复制日志文件。 |
| 64 | 构造经过上述检查后才能输出的 audit。 |
| 65 | 保存检查状态。 |
| 66 | 保存路径和重新计算的 SHA。 |
| 67 | 保存文件字节数和权重元素数。 |
| 68 | 保存完成 step 及具有 Adam 状态的参数张量个数。 |
| 69 | 保存 Adam steps 和权重有限值结论。 |
| 70 | 保存源码与原始日志核对结果。 |
| 71 | 区分原训练设备 MPS 与核验设备 CPU。 |
| 72 | 保存报告类型。 |
| 73 | 结束 audit 字典。 |
| 74 | 保存机器可读核验结果。 |
| 75 | 保存命令文本。 |
| 76 | 选择起始、停止、恢复后第一步、结束四个记录，JSON 美化但不改值。 |
| 77 | 读取生成结果。 |
| 78 | 开始生成静态 HTML；下面没有执行 shell 命令。 |
| 79 | 设置页面字符编码和标题。 |
| 80 | 开始内联样式。 |
| 81 | 定义网页字体、颜色与背景。 |
| 82 | 设置页面宽度和实验区块排版。 |
| 83 | 定义可换行的日志显示，避免长 SHA 或命令被裁掉。 |
| 84 | 设置提示框、图像宽度和链接颜色。 |
| 85 | 结束 CSS，开始页面主体。 |
| 86 | 训练结果区域与截图定位 ID。 |
| 87 | 展示数据集、更新次数和上下文长度。 |
| 88 | 展示实际参数数与本次固定硬件 / 预算标签。 |
| 89 | 展示初始和最终验证 CE。 |
| 90 | 说明第 5 步中断、第 100 步结束。 |
| 91 | 加载 loss SVG。 |
| 92 | 生成结果的分组标题。 |
| 93 | 转义并展示生成文本，结束结果区域。 |
| 94 | checkpoint 核验区域与截图定位 ID。 |
| 95 | 解释训练设备 / 检查设备差异并链接原始 JSON。 |
| 96 | 展示实际核验命令。 |
| 97 | 以可读格式展示 audit，结束核验区域。 |
| 98 | 命令区域，标明工作目录与 stdout 重定向。 |
| 99 | 说明不可覆盖已有实验，并提供纯文本命令。 |
| 100 | 转义显示复现命令。 |
| 101 | 原始日志展示区域与截图定位 ID。 |
| 102 | 介绍 JSON 日志视图。 |
| 103 | 链接原始完整两段 stdout。 |
| 104 | 展示四个更新记录，结束页面和模板字符串。 |
| 105 | 写静态 HTML 页面。 |
| 106 | 将 audit 输出到 stdout，供命令重定向保存。 |
| 109 | 只有直接执行模块时才启动脚本。 |
| 110 | 调用入口。 |

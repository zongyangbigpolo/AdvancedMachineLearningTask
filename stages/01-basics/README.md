# 第一阶段：把语言模型当作一个可验证的软件系统

Week 1 已补充[逐行代码讲解与七天任务表](../../docs/week1/README.md)，并完成[98M 模型的 100 步 Mac 教学实验](../../reports/week1/README.md)。以下正文中的 20,000 步配置仍属于尚未运行的大预算实验；不要混淆两者。

## 先建立工程映射

你已经会写复杂系统，不需要重新学习循环和类。需要建立的是这条新链路：

```text
UTF-8 文档 → byte BPE → token stream → (x, y=x右移一位)
  → decoder-only Transformer → 每个位置的词表 logits
  → next-token cross entropy → backward → AdamW → checkpoint
  → 自回归生成
```

与业务工程的关键区别：这里的“正确”分三层。**数学正确**看形状、因果遮罩、梯度；**训练正确**看目标对齐、数据无泄漏、loss 能下降、断点可恢复；**模型有效**看未见过的数据和任务上的泛化。小语料过拟合只证明前两层的一部分，不证明第三层。

## 建议按这个顺序读和动手

| 顺序 | 读代码 | 必须能够独立解释 / 完成的小练习 |
| --- | --- | --- |
| 1 | `aml/tokenizer.py` | 手算 `banana` 的相邻 pair 频数和非重叠 merge；解释为什么 256 个 byte 能表示任意 UTF-8 文本 |
| 2 | `aml/data.py` | 检查一批 `x[:,1:] == y[:,:-1]`；从二进制解码样本；区分文档边界与分块边界 |
| 3 | `aml/model.py` 的 `Linear`、`RMSNorm`、`rope` | 为每步写下输入输出维度；解释为什么 RoPE 不额外引入可训练位置参数 |
| 4 | `Attention`、`Block`、`TransformerLM` | 手算一个 3-token 的 causal mask；改变未来 token，确认过去 logits 不变 |
| 5 | `cross_entropy`、`aml/train.py` | 解释为什么先减最大值；故意去掉 target shift 观察错误目标；理解梯度累积为何除以累积次数 |
| 6 | `tests/test_basics.py`、`aml/generate.py` | 过拟合单 batch；解释 teacher forcing 与生成时逐 token 反馈的差异；跑中断与恢复对照 |
| 7 | `configs/100m.json` | 用公式核对参数和训练 token 预算，然后才申请 GPU 跑正式实验 |

第一轮学习不要直接上 100M。先沿着数据流加断点，打印 token、形状和 loss；能在纸上说清每一步再扩大规模。

## Tokenizer：不是词典查表，而是可逆编码

基础词表是 0–255 的 UTF-8 byte 加 `EOS=256`，merge ID 从 257 开始。先按 Unicode 字母串、数字串、空白串、标点串切分，再统计各段内部相邻 pair；每轮选最高频 pair，平局用 ID 字典序保证确定性。编码必须按学习到的 merge **顺序**应用，不是贪心选最长字符串。

第一道手算题只用一个文档 `banana`：

```text
bytes:       [98, 97, 110, 97, 110, 97]
pair counts: (98,97):1, (97,110):2, (110,97):2
first merge: (97,110) -> 257          平局按 pair 字典序
after merge: [98, 257, 257, 97]      对应 b + an + an + a
```

注意，`(a,a)` 在 `aaa` 中有两个相邻出现，但一次非重叠替换只能得到 `[aa,a]`，不能让中间的 `a` 被消费两次。merge 频数和实际替换次数不总相等，这是自己写 BPE 时常见的第一个 bug。

`decode(encode(text)) == text` 必须对中文、emoji、组合音标、换行与空串成立。EOS 不从输入里的某个魔法字符串自动识别，避免普通文本注入特殊 token；文档结束由数据准备显式添加。

这里的实现是易读的频数重算基线，训练复杂度较高，不是 tiktoken 的精确正则或生产 tokenizer。CLI 默认只取训练集前 200 万字符训练词表；不要直接把整个 dump 喂进内存。正式实验应按语言 / 来源分层抽样，记录抽样规则，检查 bytes/token 与不同语言压缩率。短语料可能学不满目标词表，CLI 会输出实际词表大小；模型允许有未使用的词表槽位。

## Transformer：形状就是接口契约

用 `B` 表示 batch、`T` 表示 context、`D` 表示隐藏宽度、`H` 表示头数、`Dh=D/H`：

```text
ids                  [B,T]
embedding            [B,T,D]
Q,K,V                [B,H,T,Dh]
QKᵀ / sqrt(Dh)       [B,H,T,T]
causal softmax       [B,H,T,T]   上三角置 -inf，不屏蔽自身
attention output     [B,T,D]
SwiGLU intermediate  [B,T,F]
logits               [B,T,V]
targets              [B,T]
```

最小的 next-token 例子：原始窗口为 `[10,20,30,256]`，则 `x=[10,20,30]`、`y=[20,30,256]`。位置 0 根据 token `10` 预测 `20`，位置 1 根据 `10,20` 预测 `30`，位置 2 预测文档结束。如果不右移，模型学的是复制输入，不是语言建模；如果不遮住未来，训练时可以偷看答案，生成时却拿不到答案。

每个 block：

```text
h = x + Attention(RMSNorm(x))
y = h + down(silu(gate(RMSNorm(h))) * up(RMSNorm(h)))
```

Q、K 使用 RoPE，V 不用；无 bias、无 dropout。输出投影与输入 embedding 共享参数。残差输出初始化按层数缩放。核心公式：

```text
RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
Attention(Q,K,V) = softmax(QKᵀ / sqrt(Dh) + causal_mask)V
CE = mean(logsumexp(logits) - logits[target])
```

训练时一个前向同时预测每个位置的下一个 token；生成时只用末尾位置分布，再把采样结果送回模型。当前无 KV cache，超过 context 时截取最近的窗口并重算位置；这是基线而非高效推理。

## 98M 参数的账算清楚

默认配置：`V=8192, D=768, L=12, H=12, F=2304, T=512`，输入输出权重共享。

```text
P = VD + L(4D² + 3DF + 2D) + D
  = 98,323,200
```

“0.1B”指约一亿参数，不等于训练一亿个 token。默认有效 batch 是 `2 × 512 × 16 = 16,384 tokens/update`；20,000 updates 共 **327,680,000 采样 token**，不是 3.28 亿条独立样本，也不保证这些 token 唯一。常用 `20 tokens/parameter` 只能作为预算启发式，对这个模型约 19.7 亿 tokens，向上取整需 120,024 updates；不是任何语料与模型上的最优定律证明。

目前 `20,000` steps 是可运行的实验配置，不代表充分训练或论文级效果。样本按随机连续窗口有放回抽取，EOS 是普通可学习边界，attention 允许跨文档；这是明确的 packing 选择，不是 block-diagonal 文档隔离。

FP32 参数约 393 MB，梯度再约 393 MB，Adam 两份 moment 约 787 MB；仅这几项就约 1.57 GB，**不包含**激活、注意力矩阵、临时张量、框架开销。显式 Attention 的 `[B,H,T,T]` 内存随 `T²` 增长；不要把静态参数体积当显存需求。先在目标 GPU 上短跑并监控显存、step time，再决定 micro-batch；不承诺某个设备一定放得下。

## 从微型 Demo 进入真实语料

先运行根 README 的 smoke。然后自行准备已获使用授权的 UTF-8 数据：

```text
data/raw/train.txt    每行一个非空文档
data/raw/val.txt      独立的 held-out 文档
```

按文档 / 来源先划分，再去重，最后训练 tokenizer；验证集不能进入词表训练、超参训练目标或训练 token 流。工具会拒绝完全相同的两个文件，但**不能检测跨文件的局部重复、近重复或来源泄漏**，这需要数据审计与第三阶段实现。换行在此定义为文档边界；有多行文档时先转换格式，不要无意识地逐行切散。

```bash
python -m aml.tokenizer --input data/raw/train.txt \
  --output data/tokenizer-8192.json --vocab-size 8192 --max-chars 2000000
python -m aml.data --train data/raw/train.txt --val data/raw/val.txt \
  --tokenizer data/tokenizer-8192.json --output data/pretrain

# 先跑到 20 步检查配置与资源，不改变原定学习率调度
python -m aml.train --config configs/100m.json --data data/pretrain \
  --output runs/100m-baseline --device cuda --stop-after 20

# 使用相同配置、数据、设备类型，恢复至配置的总步数
python -m aml.train --config configs/100m.json --data data/pretrain \
  --output runs/100m-baseline --device cuda --resume

python -m aml.generate --checkpoint runs/100m-baseline/latest.pt \
  --prompt "Once upon a time" --max-new-tokens 100 --device cuda
```

Mac 可以选择 `--device mps`；CPU 可以用于正确性验证。Triton / CUDA 多卡不在 macOS 本机实现。工具不会下载大语料、租用 GPU 或自动启动全量训练；需要在已经准备好的硬件上显式运行。

部分 PyTorch 版本在未安装可选 NumPy 时会输出 `Failed to initialize NumPy` 警告；本实现没有使用 Tensor / NumPy 桥接，分词和 token 文件读写也不依赖 NumPy，不需要为这一警告改变核心实现。

## 训练产物与验收

`manifest.json` 保存完整配置、参数量、设备、PyTorch / Python 版本、源码摘要与数据指纹；`metrics.jsonl` 逐步保存训练 loss、学习率、梯度范数、token 数和吞吐，定期保存固定验证窗口的平均 loss。吞吐只包含采样、训练、同步，不含评估、checkpoint 与 JSON 日志开销；不能直接当作端到端吞吐。

`latest.pt` 原子替换，保存模型、优化器、step、CPU / 设备 RNG 和采样器状态；恢复要求原配置与数据指纹不变。CPU 单线程的中断 / 连续训练测试要求逐参数完全相同；跨设备、不同库版本不保证位级可复现。这里使用 PyTorch AdamW，当前阶段没有声称优化器也是手写的。不要加载不可信来源的 checkpoint。

| 验收项 | 判定方法 |
| --- | --- |
| 分词正确 | 多语言 byte round-trip、确定性、保存重载、EOS 边界 |
| 模型正确 | Attention 前向与输入梯度对照 PyTorch oracle；未来输入不影响过去 logits |
| 优化目标正确 | 手写 CE 与 oracle 的值 / 梯度一致；单 batch loss 降到初值的 10% 以下 |
| 状态完整 | 小模型 CPU 连续训练和分段恢复参数完全一致 |
| 参数规模正确 | 实际实例化模型的 `parameter_count()` 在 95M–105M，且等于配置公式 |
| **0.1B 教学实验完成** | Mac 上实际完成 98M 模型、100 步 / 12,800 token、中断恢复、评估与生成；证据见 `reports/week1/`，不代表充分预训练 |
| **大预算预训练，尚待运行** | 在足量审计语料上完成 `configs/100m.json` 声明的预算，并进行更完整的泛化与能力评估 |

验证 loss 是固定随机窗口的抽样估计，不是全验证集 perplexity。不同 tokenizer 的 token CE 不能直接横比；需要统一评估协议或采用规范化的 bits/byte。不得用 smoke 的 loss 或自动化测试通过替代 0.1B 实验结果。

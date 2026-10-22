# Mac 上的 0.1B 教学实验

## 固定预算

| 项目 | 设置 |
| --- | --- |
| 模型 | 98,323,200 可训练参数；12 层、D=768、F=2304、12 heads、8192 词表槽位 |
| 运行设备 | 当前 Apple M4、10 GPU cores、16 GiB unified memory，PyTorch MPS |
| 精度 | FP32，不调用现成 SDPA 或 Transformer |
| 上下文 | 128；缩短 context 不改变参数量，因为使用无参数 RoPE |
| 训练预算 | 100 optimizer updates、batch=1、grad_accum=1，共 12,800 采样 token |
| 数据预算 | TinyStories 上游 train 前 512 个完整故事、validation 前 128 个 |
| 词表 | 只在训练原文学习，最大 8192；本次学到 5696 个实际 token |
| 验证 | 每 10 步与本次停止点评估；固定 4 个 128-token 验证窗口，共 512 预测位置，窗口可能重叠 |
| 恢复演示 | 先停在 step 5，再恢复到 step 100；学习率仍按总共 100 步调度 |
| 费用 | 不租用外部 GPU，不创建付费服务；使用本地机器 |

本次预算为 100 次更新、12,800 个采样 token，约 0.00013 token/parameter。验证损失从 8.83 降至 3.51，生成文本则出现高频词重复；这提示下一步应增加语料与训练预算，并将损失变化和生成质量分开分析。

## 从干净目录重放

先在仓库根目录执行 `cd stages/01-basics`，以下所有命令都在第一阶段目录运行：

```bash
uv sync --locked --python 3.12
source .venv/bin/activate
python -m unittest discover -s tests -v

python -m aml.tinystories --output data/week1-raw \
  --train-rows 512 --val-rows 128
python -m aml.tokenizer --input data/week1-raw/train.txt \
  --output data/week1-tokenizer.json --vocab-size 8192
python -m aml.data --train data/week1-raw/train.txt \
  --val data/week1-raw/val.txt --tokenizer data/week1-tokenizer.json \
  --output data/week1-tokens

python -m aml.train --config configs/100m-mac.json \
  --data data/week1-tokens --output runs/week1-100m-mac \
  --device mps --stop-after 5
python -m aml.train --config configs/100m-mac.json \
  --data data/week1-tokens --output runs/week1-100m-mac \
  --device mps --resume
python -m aml.generate --checkpoint runs/week1-100m-mac/latest.pt \
  --prompt "Once upon a time" --temperature 0 --max-new-tokens 64 --device mps
```

非空数据 / 实验目录不会覆盖；重跑时给所有命令换一组新的目录名。不要把 `--resume` 理解为任意换配置继续训练；它要求同配置、数据指纹和设备类型。

## 数据授权与复现限制

来源：[roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)，作者 Ronen Eldan、Yuanzhi Li；许可为 [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/)。下载器保留署名、许可链接和空白规范化说明，每行保存一个完整故事。小数据包见第一阶段 `artifacts/`，完整 checkpoint 见同仓库 Releases。

API 返回页未锁定数据集 revision。`source.json` 保存请求 URL、页响应 SHA 和最终文本 SHA；远端数据变化时按不同数据版本记录。质量检查覆盖精确重复和跨集重叠；语义近重复、模板泄漏及随机抽样对照尚未展开。

## 现场 Demo

演示顺序：BPE 编解码往返、训练曲线与预算、运行配置与指标、加载权重生成。结果分析关注重复输出、上下文长度、验证采样和数据规模。

模型权重可从同仓库的 [Releases](https://github.com/zongyangbigpolo/AdvancedMachineLearningTask/releases) 下载；小数据包随第一阶段提交。checkpoint 包含 Adam 状态，单份约 1.18 GB，因此使用 Release 分发。

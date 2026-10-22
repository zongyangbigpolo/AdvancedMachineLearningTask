# 98M 模型训练与断点恢复

## 固定预算

| 项目 | 设置 |
| --- | --- |
| 模型 | 98,323,200 可训练参数；12 层、D=768、F=2304、12 heads、8192 词表槽位 |
| 运行设备 | Apple M4、10 GPU cores、16 GiB unified memory，PyTorch MPS |
| 精度 | FP32 |
| 上下文 | 128；缩短 context 不改变参数量，因为使用无参数 RoPE |
| 训练预算 | 100 optimizer updates、batch=1、grad_accum=1，共 12,800 采样 token |
| 数据预算 | TinyStories 上游 train 前 512 个完整故事、validation 前 128 个 |
| 词表 | 只在训练原文学习，最大 8192；本次学到 5696 个实际 token |
| 验证 | 每 10 步与本次停止点评估；固定 4 个 128-token 验证窗口，共 512 预测位置，窗口可能重叠 |
| 恢复演示 | 先停在 step 5，再恢复到 step 100；学习率仍按总共 100 步调度 |

本次预算为 100 次更新、12,800 个采样 token，约 0.00013 token/parameter。验证损失从 8.83 降至 3.51，生成文本则出现高频词重复；这提示下一步应增加语料与训练预算，并将损失变化和生成质量分开分析。

## 复现命令

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

数据与实验输出目录须为空。`--resume` 要求配置、数据指纹和设备类型与快照一致。

## 数据授权与复现限制

来源：[roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)，作者 Ronen Eldan、Yuanzhi Li；许可为 [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/)。下载器保留署名、许可链接和空白规范化说明，每行保存一个完整故事。小数据包见第一阶段 `artifacts/`，完整 checkpoint 见同仓库 Releases。

API 返回页未锁定数据集 revision。`source.json` 保存请求 URL、页响应 SHA 和最终文本 SHA；远端数据变化时按不同数据版本记录。质量检查覆盖精确重复和跨集重叠；语义近重复、模板泄漏及随机抽样对照尚未展开。

## 模型与数据

[训练快照](https://github.com/zongyangbigpolo/AdvancedMachineLearningTask/releases/tag/stage1-100-updates)包含模型权重、Adam 状态和随机状态，大小约 1.18 GB。[数据包](../../artifacts/training-data.zip)包含训练与验证文本、分词器、编码文件及元数据。

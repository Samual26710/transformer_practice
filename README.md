# Transformer from Scratch

一个从零手写 Transformer 的练习项目，参考「Attention Is All You Need」论文与常见教程实现，用于机器翻译（英 → 意）。

## 文件结构

- `model.py` — Transformer 核心组件
  - `InputEmbeddings` / `PositionalEncoding`：词嵌入与位置编码
  - `MultiHeadAttention`：多头自注意力
  - `FeedForward` / `LayerNormalization` / `ResidualConnection`：前馈网络、层归一化、残差连接
  - `EncoderBlock` / `Encoder`、`DecoderBlock` / `Decoder`：编码器与解码器
  - `ProjectionLayer` / `Transformer` / `build_transformer`：投影层与整体组装
- `dataset.py` — `BilingualDataset` 数据集类，负责 tokenize、添加 SOS/EOS/PAD、构造 encoder/decoder 掩码，以及 `causal_mask` 因果掩码
- `config.py` — 训练超参数与路径配置
- `train.py` — 训练主流程
  - 构建 / 加载 WordLevel tokenizer
  - 加载 `opus_books` 数据集并划分 train / val
  - 训练循环：Noam 学习率调度、tensorboard 记录
  - `greedy_decode`：贪心解码做推理
  - `run_validation`：在验证集上生成翻译并用 BLEU 评估
  - 每个 epoch 保存检查点

## 依赖

```bash
pip install torch torchmetrics datasets tokenizers tensorboard tqdm
```

## 运行

```bash
python train.py
```

训练权重保存在 `weights/`，tensorboard 日志在 `runs/`。

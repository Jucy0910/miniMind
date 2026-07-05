# my_minimind

这里是我们手搓 MiniMind 的实现目录。

第一阶段只搭项目骨架。目标是在实现 Transformer 细节前，先把代码结构、模块边界和入口约定定下来。

模型默认配置会尽量对齐 `original_project/minimind-3/config.json` 中的 MiniMind-3 Dense 配置：

```text
vocab_size=6400
hidden_size=768
num_hidden_layers=8
num_attention_heads=8
num_key_value_heads=4
head_dim=96
intermediate_size=2432
max_position_embeddings=32768
rope_theta=1000000.0
```

## 目录结构

```text
my_minimind/
  configs/          模型配置和训练配置
  data/             tokenizer 加载和数据集代码
  model/            模型接口和后续 Transformer 实现
  train/            训练入口
  eval/             生成和评测入口
  tests/            小型冒烟测试
```

真实训练语料不要放进 `my_minimind/data/`，推荐放在仓库根目录的 `datasets/`：

```text
datasets/
  pretrain_t2t_mini.jsonl
```

这样 `my_minimind/` 只放代码，`datasets/` 只放本地大数据文件。

## 第一阶段冒烟检查

在仓库根目录执行：

```bash
python -m py_compile \
  my_minimind/configs/model_config.py \
  my_minimind/data/tokenizer.py \
  my_minimind/data/dataset.py \
  my_minimind/model/minimind.py \
  my_minimind/train/train_pretrain.py \
  my_minimind/eval/generate.py
```

打印默认模型配置：

```bash
python -m my_minimind.train.train_pretrain --dry_run
```

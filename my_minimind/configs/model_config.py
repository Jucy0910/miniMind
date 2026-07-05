# dataclass 用来少写配置类的 __init__ 等样板代码。
# asdict 可以把 dataclass 对象转换成普通 dict，方便保存成 JSON。
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class MiniMindConfig:
    """第一版 Dense Causal LM 的模型配置。"""

    # 词表大小，也就是 tokenizer 一共有多少种 token。
    # 模型最后一层 logits 的最后一维也会等于 vocab_size。
    vocab_size: int = 6400

    # Transformer 中每个 token 向量的维度。
    # hidden_size 越大，模型表达能力越强，但显存和计算量也越高。
    hidden_size: int = 768

    # TransformerBlock 的层数。
    # 每一层通常包含 Attention + MLP。
    num_hidden_layers: int = 8

    # Query 的注意力头数量。
    # hidden_size 必须能整除 num_attention_heads。
    num_attention_heads: int = 8

    # Key/Value 的注意力头数量。
    # 小于 num_attention_heads 时就是 GQA，能减少 KV cache 和计算开销。
    num_key_value_heads: int = 4

    # MLP 中间层维度。
    # 通常会大于 hidden_size，MiniMind 原项目里也是单独配置。
    intermediate_size: int = 2432

    # 模型支持的最大上下文长度。
    # RoPE 位置编码缓存和训练截断长度都会参考这个值。
    max_position_embeddings: int = 32768

    # dropout 概率。
    # 小模型和大规模训练里经常设为 0，先保留这个配置位。
    dropout: float = 0.0

    # RMSNorm 的数值稳定项，防止除以非常小的数。
    rms_norm_eps: float = 1e-6

    # RoPE 频率基数。
    # MiniMind 原项目使用较大的 1e6，这里保持相近默认值。
    rope_theta: float = 1_000_000.0

    # tokenizer 中特殊 token 的 id。
    # 这三个 id 会影响数据集拼接、padding 和 loss mask。
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    # 是否让输入 embedding 和输出 lm_head 共用同一份权重。
    # GPT 类模型常用这个技巧，可以减少参数量。
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        # 每个注意力头分到的向量维度。
        # 例如 hidden_size=512, num_attention_heads=8，则 head_dim=64。
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        return self.hidden_size // self.num_attention_heads

    def to_dict(self) -> dict[str, Any]:
        # asdict 只会保存 dataclass 字段。
        # head_dim 是 @property 计算属性，不会自动出现在 asdict 结果里，所以手动补上。
        data = asdict(self)
        data["head_dim"] = self.head_dim
        return data

    @classmethod
    def from_json(cls, path: str | Path) -> "MiniMindConfig":
        # 从 JSON 文件恢复配置。
        # 这里允许 JSON 里有 head_dim，但加载时会丢弃它，因为 head_dim 应由配置自动计算。
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        data.pop("head_dim", None)
        return cls(**data)

    def save_json(self, path: str | Path) -> None:
        # 保存配置时先创建父目录，避免目录不存在导致写文件失败。
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            # ensure_ascii=False 可以让中文正常写入 JSON，而不是变成 \uXXXX。
            # indent=2 让 JSON 更适合人工阅读。
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            f.write("\n")


@dataclass
class TrainConfig:
    """早期小实验使用的默认训练配置。"""

    # 单条样本最终截断或 padding 到的 token 长度。
    max_seq_len: int = 512

    # 每个 step 读入多少条样本。
    batch_size: int = 8

    # 完整遍历训练集的次数。
    epochs: int = 1

    # AdamW 的初始学习率。
    learning_rate: float = 5e-4

    # AdamW 的权重衰减，常用于抑制过拟合。
    weight_decay: float = 0.1

    # 梯度裁剪阈值，避免梯度爆炸。
    grad_clip: float = 1.0

    # 梯度累积步数。
    # 显存不够时可以用多个小 batch 累积成一个等效大 batch。
    accumulation_steps: int = 1

    # DataLoader 的工作进程数。
    # 第一阶段默认 0，方便调试时看到完整报错栈。
    num_workers: int = 0

    # 每隔多少 step 打印一次日志。
    log_interval: int = 10

    # 每隔多少 step 保存一次 checkpoint。
    save_interval: int = 200

    # 随机种子，用来尽量复现实验。
    seed: int = 42

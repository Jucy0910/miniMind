from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any


@dataclass
class MiniMindConfig:
    """MiniMind 模型结构配置。
    这个类只描述“模型长什么样”，不负责训练超参数。
    默认值对齐原始 MiniMind dense 模型，后续实现 RMSNorm、RoPE、Attention、
    MLP 和 TransformerBlock 时都会从这里读取结构参数。
    """
    # 模型类型名称。这里使用 minimind，表示这是我们手写的 MiniMind 实现。
    model_type: str = "my_minimind"

    # tokenizer 的词表大小。
    # embedding 输入表和 lm_head 输出表都会使用这个大小。
    vocab_size: int = 6400

    # Transformer 内部每个 token 的向量维度。
    hidden_size: int = 768

    # TransformerBlock 的层数。
    num_hidden_layers: int = 8

    # Query 的注意力头数。
    num_attention_heads: int = 8

    # Key/Value 的注意力头数。
    # 这里是 4，小于 query head 的 8，所以是 GQA。
    num_key_value_heads: int | None = 4

    # 每个 attention head 的维度。
    # 原始 MiniMind-3 是 96，也就是 768 / 8。
    head_dim: int | None = None

    # FFN 中间层维度。
    # 如果不手动传入，就按原始 MiniMind 公式计算：ceil(hidden_size * pi / 64) * 64。
    intermediate_size: int | None = None

    # FFN 激活函数。silu 对应后面要实现的 SwiGLU 结构。
    hidden_act: str = "silu"

    # 模型支持的最大上下文长度。
    # RoPE 的 cos/sin 缓存会按照这个长度预计算。
    max_position_embeddings: int = 32768

    # RoPE 的频率基数。原始 MiniMind 使用 1e6。
    rope_theta: float = 1_000_000.0

    # 推理扩展上下文时可开启 YaRN rope scaling。
    # 训练阶段默认不开启。
    inference_rope_scaling: bool = False
    rope_scaling: dict[str, Any] | None = None

    # RMSNorm 的数值稳定项。
    rms_norm_eps: float = 1e-6

    # dropout 概率。原始 dense 配置默认是 0。
    dropout: float = 0.0

    # 后面 Attention 可以根据这个开关使用 PyTorch 的高效 attention 实现。
    flash_attn: bool = True

    # 输入 embedding 和输出 lm_head 是否共享权重。
    tie_word_embeddings: bool = True

    # tokenizer 特殊 token id。
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    # 参数初始化标准差，后面初始化模型权重时会用到。
    initializer_range: float = 0.02

    # 推理生成时是否默认使用 KV cache。
    use_cache: bool = True

    # MoE 相关字段先保留配置位，但 dense 模型默认不会使用。
    use_moe: bool = False
    num_experts: int = 4
    num_experts_per_tok: int = 1
    moe_intermediate_size: int | None = None
    norm_topk_prob: bool = True
    router_aux_loss_coef: float = 5e-4

    def __post_init__(self) -> None:
        """补齐派生字段，并检查配置是否合法。"""

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size 必须能被 num_attention_heads 整除")

        expected_head_dim = self.hidden_size // self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = expected_head_dim
        elif self.head_dim != expected_head_dim:
            raise ValueError(
                "head_dim 必须等于 hidden_size // num_attention_heads，"
                f"当前 head_dim={self.head_dim}, 期望 {expected_head_dim}"
            )

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads 必须能被 num_key_value_heads 整除")

        if self.intermediate_size is None:
            self.intermediate_size = math.ceil(self.hidden_size * math.pi / 64) * 64

        if self.moe_intermediate_size is None:
            self.moe_intermediate_size = self.intermediate_size

        if self.inference_rope_scaling and self.rope_scaling is None:
            self.rope_scaling = {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MiniMindConfig":
        """从字典构造配置。
        原始模型配置里可能有 architectures、transformers_version 等 Hugging Face 字段。
        这些字段不是我们手写模型当前需要的内容，所以这里自动忽略。
        """

        valid_keys = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str | Path) -> "MiniMindConfig":
        """从 JSON 文件读取配置。"""

        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """转换成普通字典，方便打印、保存和调试。"""

        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        """保存配置到 JSON 文件。"""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            f.write("\n")

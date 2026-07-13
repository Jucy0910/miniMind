from __future__ import annotations

import torch
from torch import nn

from .config import MiniMindConfig
from .layers import MiniMindBlock, RMSNorm, RotaryEmbedding


# 单层 KV cache：第一个张量是 Key，第二个张量是 Value。
# 两者形状都是 [batch, cache_len, num_key_value_heads, head_dim]。
KVCache = tuple[torch.Tensor, torch.Tensor]

# 完整模型有多层 Transformer，所以每一层都需要保存自己的 KV cache。
PastKeyValues = tuple[KVCache, ...]


class MiniMindModel(nn.Module):
    """MiniMind 基础模型主干。
    这个类负责：
    input_ids -> Token Embedding -> 多层 MiniMindBlock -> Final RMSNorm。
    它输出 hidden_states，不负责投影到词表，也不负责计算语言模型 loss。
    LM Head 和 loss 会在后续 MiniMindForCausalLM 中实现。
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        # 1. Token Embedding：把离散 token id 转换为 hidden_size 维向量。
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.dropout)
        # 2. 整个模型只创建一份 RoPE cos/sin 缓存。
        #    每一层 Attention 都使用同一个 RotaryEmbedding，避免重复占用内存。
        self.rotary_emb = RotaryEmbedding(
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        # 3. 按配置创建多层 Transformer Block。
        self.layers = nn.ModuleList(
            [MiniMindBlock(config) for _ in range(config.num_hidden_layers)]
        )
        # 4. 所有 Transformer Block 结束后，再做一次最终 RMSNorm。
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 5. 按 MiniMindConfig.initializer_range 初始化可训练权重。
        self.apply(self._init_weights)
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, PastKeyValues | None]:
        """执行完整 MiniMind 主干前向传播。
        input_ids 形状：[batch, seq_len]。
        返回 hidden_states 形状：[batch, seq_len, hidden_size]。
        use_cache=True 时，第二个返回值包含每一层的新 KV cache。
        """
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids 必须是二维张量 [batch, seq_len]，"
                f"当前形状为 {tuple(input_ids.shape)}"
            )
        # 没有历史 cache 时，为每一层准备一个 None 占位。
        if past_key_values is None:
            layer_past_key_values: tuple[KVCache | None, ...] = tuple(
                None for _ in range(self.num_hidden_layers)
            )
            start_pos = 0
        else:
            if len(past_key_values) != self.num_hidden_layers:
                raise ValueError(
                    "past_key_values 的层数必须等于 num_hidden_layers，"
                    f"当前 cache 层数={len(past_key_values)}, "
                    f"模型层数={self.num_hidden_layers}"
                )
            layer_past_key_values = past_key_values
            # 所有层 cache 的序列长度应该一致，取第一层长度作为当前 RoPE 起始位置。
            start_pos = past_key_values[0][0].shape[1]

        # 1. token id -> token embedding。
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.embed_dropout(hidden_states)

        # use_cache=True 时，把每层新产生的 KV cache 收集起来。
        present_key_values: list[KVCache] = []

        # 2. hidden_states 依次通过每一个 Transformer Block。
        for layer, past_key_value in zip(self.layers, layer_past_key_values):
            hidden_states, present_key_value = layer(
                hidden_states=hidden_states,
                rotary_emb=self.rotary_emb,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                start_pos=start_pos,
            )

            if use_cache:
                # use_cache=True 时，Block 一定会返回当前层的新 cache。
                if present_key_value is None:
                    raise RuntimeError("use_cache=True，但 Transformer Block 没有返回 KV cache")
                present_key_values.append(present_key_value)

        # 3. 多层 Block 结束后做最终归一化。
        hidden_states = self.norm(hidden_states)

        if use_cache:
            return hidden_states, tuple(present_key_values)
        return hidden_states, None

    def _init_weights(self, module: nn.Module) -> None:
        """初始化 Linear 和 Embedding 的可训练权重。"""

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

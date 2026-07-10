from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import MiniMindConfig


class RMSNorm(nn.Module):
    """RMSNorm 归一化层。
    RMSNorm 只按最后一维计算均方根，不像 LayerNorm 那样减均值。
    对语言模型来说，最后一维通常就是 hidden_size。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        # eps 用来避免除以 0，尤其是在输入全 0 或数值很小时。
        self.eps = eps

        # weight 是可训练缩放参数。
        # 初始全 1，表示一开始只做归一化，不额外改变每个通道的尺度。
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """对 hidden_states 的最后一维做 RMS 归一化。"""

        # 归一化时转成 float32 更稳定。
        # 如果输入是 float16/bfloat16，最后再转回原 dtype，和原始 MiniMind 实现一致。
        input_dtype = hidden_states.dtype
        hidden_states_float = hidden_states.float()

        # variance 这里不是严格统计学里的方差，而是 x^2 的均值。
        # 形状从 [..., hidden_size] 变成 [..., 1]，方便广播回每个 hidden 维度。
        variance = hidden_states_float.pow(2).mean(dim=-1, keepdim=True)

        # rsqrt(x) 等价于 1 / sqrt(x)，比先 sqrt 再除更直接。
        normalized = hidden_states_float * torch.rsqrt(variance + self.eps)

        # weight 会按最后一维广播，例如 [hidden_size] 广播到 [batch, seq, hidden_size]。
        return (self.weight * normalized).to(dtype=input_dtype)


class RotaryEmbedding(nn.Module):
    """RoPE 旋转位置编码组件。

    RoPE 不像绝对位置编码那样把位置向量加到 hidden_states 上。
    它是在 Attention 内部对 query/key 做旋转，让注意力天然带上相对位置信息。
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float = 1_000_000.0,
        rope_scaling: dict | None = None,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE 要求 head_dim 必须是偶数")
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        # ===== 初始化阶段：一次性构建完整 cos/sin 缓存 =====
        # 1. RoPE 每两个维度构成一个旋转平面，所以只需要 head_dim // 2 个频率。
        #    例如 head_dim=8 时，freq_indices=[0, 2, 4, 6]，对应 4 个旋转频率。
        freq_indices = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        freqs = 1.0 / (self.rope_theta ** (freq_indices / self.head_dim))

        # 2. attention_factor 是 YaRN 等长上下文扩展里的额外缩放系数。
        #    训练默认不缩放，保持原始 MiniMind dense 配置。
        attention_factor = 1.0

        if self.rope_scaling is not None:
            # 这个分支对齐原始 MiniMind 的 YaRN 风格缩放逻辑。
            # 当前训练默认 rope_scaling=None，所以可以先理解成预留扩展。
            original_max = self.rope_scaling.get("original_max_position_embeddings", 2048)
            factor = self.rope_scaling.get("factor", 16)
            beta_fast = self.rope_scaling.get("beta_fast", 32.0)
            beta_slow = self.rope_scaling.get("beta_slow", 1.0)
            attention_factor = self.rope_scaling.get("attention_factor", 1.0)

            if self.max_position_embeddings / original_max > 1.0:
                low = max(
                    math.floor(
                        self.head_dim
                        * math.log(original_max / (beta_fast * 2 * math.pi))
                        / (2 * math.log(self.rope_theta))
                    ),
                    0,
                )
                high = min(
                    math.ceil(
                        self.head_dim
                        * math.log(original_max / (beta_slow * 2 * math.pi))
                        / (2 * math.log(self.rope_theta))
                    ),
                    self.head_dim // 2 - 1,
                )

                # ramp 从 0 平滑过渡到 1，用来让不同频率分量逐步缩放。
                ramp = torch.arange(self.head_dim // 2, dtype=torch.float32)
                ramp = torch.clamp((ramp - low) / max(high - low, 0.001), 0, 1)
                freqs = freqs * (1 - ramp + ramp / factor)

        # 3. positions 是每个 token 的位置编号：[0, 1, 2, ...]。
        positions = torch.arange(self.max_position_embeddings, dtype=torch.float32)

        # 4. 外积得到角度表：角度 = 位置编号 * 频率。
        #    形状从 [max_position_embeddings] 和 [head_dim // 2]
        #    变成 [max_position_embeddings, head_dim // 2]。
        angles = torch.outer(positions, freqs)

        # 5. 每个旋转平面的一对维度使用同一个角度。
        #    这里把 [head_dim // 2] 复制成 [head_dim]，便于后面和 q/k 逐元素相乘。
        cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)
        sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)

        # cos/sin 是固定缓存，不是可训练参数。
        # register_buffer 会让它们跟随模块一起迁移到 GPU，但不会进入优化器。
        self.register_buffer("cos_cached", cos * attention_factor, persistent=False)
        self.register_buffer("sin_cached", sin * attention_factor, persistent=False)

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """对 query/key 应用 RoPE。

        query_states 典型形状：[batch, seq_len, num_heads, head_dim]。
        key_states 典型形状：[batch, seq_len, num_key_value_heads, head_dim]。
        start_pos 用于后续 KV cache 推理，训练时通常是 0。
        """

        # 1. 根据当前输入长度，从缓存中切出本次 forward 需要的位置片段。
        seq_len = query_states.shape[1]
        end_pos = start_pos + seq_len
        if end_pos > self.max_position_embeddings:
            raise ValueError(
                "RoPE 位置超过 max_position_embeddings："
                f"end_pos={end_pos}, max_position_embeddings={self.max_position_embeddings}"
            )

        cos = self.cos_cached[start_pos:end_pos].to(device=query_states.device)
        sin = self.sin_cached[start_pos:end_pos].to(device=query_states.device)

        # 2. q/k 布局是 [batch, seq, heads, head_dim]。
        #    cos/sin 是 [seq, head_dim]，unsqueeze 后可广播到 head 维度。
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # 3. 把最后一维拆成前后两半并旋转。
        #    [x1, x2] -> [-x2, x1]，配合 cos/sin 完成二维旋转。
        query_half = query_states.shape[-1] // 2
        query_first_half = query_states[..., :query_half]
        query_second_half = query_states[..., query_half:]
        query_rotated_half = torch.cat((-query_second_half, query_first_half), dim=-1)

        key_half = key_states.shape[-1] // 2
        key_first_half = key_states[..., :key_half]
        key_second_half = key_states[..., key_half:]
        key_rotated_half = torch.cat((-key_second_half, key_first_half), dim=-1)

        # 4. RoPE 旋转公式：x_rot = x * cos + rotate_half(x) * sin。
        query_embed = query_states * cos + query_rotated_half * sin
        key_embed = key_states * cos + key_rotated_half * sin
        return query_embed.to(dtype=query_states.dtype), key_embed.to(dtype=key_states.dtype)


class MiniMindAttention(nn.Module):
    """MiniMind 使用的因果自注意力层。

    这里实现的是 GQA：Query head 数量可以多于 Key/Value head 数量。
    例如 MiniMind 默认 Q 有 8 个头，K/V 有 4 个头。
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        # Q 投影保留完整 attention head 数。
        # K/V 投影只生成较少的 KV head，这是 GQA 降低 KV cache 的关键。
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )

        # 注意力输出先是 num_attention_heads * head_dim，再投回 hidden_size。
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )

        # 原始 MiniMind 会对每个 head 内的 q/k 再做 RMSNorm。
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: RotaryEmbedding,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """计算因果自注意力。
        hidden_states 形状：[batch, seq_len, hidden_size]。
        返回 output 形状仍然是：[batch, seq_len, hidden_size]。
        """
        batch_size, seq_len, _ = hidden_states.shape
        # 1. 从同一份 hidden_states 投影出 Q/K/V。
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 2. 拆成多头布局。
        #    Q: [batch, seq, q_heads, head_dim]
        #    K/V: [batch, seq, kv_heads, head_dim]
        query_states = query_states.view(
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.head_dim,
        )
        key_states = key_states.view(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        )
        value_states = value_states.view(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        )

        # 3. 对每个 head 内的 Q/K 做 RMSNorm，然后注入 RoPE 位置信息。
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        if past_key_value is not None and start_pos == 0:
            start_pos = past_key_value[0].shape[1]
        query_states, key_states = rotary_emb(query_states, key_states, start_pos=start_pos)

        # 4. 如果推理时传入了历史 K/V，把当前 K/V 拼到历史后面。
        #    cache 中保存的是尚未 repeat 的 KV head，这样显存最省。
        if past_key_value is not None:
            past_key_states, past_value_states = past_key_value
            key_states = torch.cat([past_key_states, key_states], dim=1)
            value_states = torch.cat([past_value_states, value_states], dim=1)

        present_key_value = (key_states, value_states) if use_cache else None

        # 5. 真正的多头注意力计算放到类内部的独立函数里。
        attn_output = self._compute_attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
        )

        # 6. 合并多头并投回 hidden_size。
        attn_output = attn_output.view(batch_size, seq_len, self.num_attention_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output, present_key_value

    def _compute_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """执行多头注意力计算。
        query_states: [batch, query_len, q_heads, head_dim]
        key_states:   [batch, key_len, kv_heads, head_dim]
        value_states: [batch, key_len, kv_heads, head_dim]
        返回形状：  [batch, query_len, q_heads, head_dim]
        """
        batch_size, query_len, _, _ = query_states.shape
        key_len = key_states.shape[1]
        # 1. GQA：K/V head 少于 Q head，需要复制到和 Q head 数一致。
        #    cache 里仍然保存较少的 KV head；只有真正算注意力时才临时展开。
        if self.num_key_value_groups > 1:
            key_states = key_states[:, :, :, None, :].expand(
                batch_size,
                key_len,
                self.num_key_value_heads,
                self.num_key_value_groups,
                self.head_dim,
            )
            key_states = key_states.reshape(
                batch_size,
                key_len,
                self.num_attention_heads,
                self.head_dim,
            )
            value_states = value_states[:, :, :, None, :].expand(
                batch_size,
                key_len,
                self.num_key_value_heads,
                self.num_key_value_groups,
                self.head_dim,
            )
            value_states = value_states.reshape(
                batch_size,
                key_len,
                self.num_attention_heads,
                self.head_dim,
            )

        # 2. 调整到注意力计算常用布局：[batch, heads, seq, head_dim]。
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # 3. QK^T 得到注意力分数，除以 sqrt(head_dim) 稳定 softmax。
        attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        # 4. 因果 mask：当前 token 不能看未来 token。
        #    如果使用 KV cache，key_len 会大于 query_len，past_len 表示历史长度。
        past_len = key_len - query_len
        query_positions = past_len + torch.arange(query_len, device=query_states.device)
        key_positions = torch.arange(key_len, device=query_states.device)
        causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        # 5. padding mask：1 表示有效 token，0 表示 padding。
        if attention_mask is not None:
            if attention_mask.shape[-1] != key_len:
                raise ValueError(
                    "attention_mask 最后一维必须等于 key_len，"
                    f"当前 attention_mask.shape={tuple(attention_mask.shape)}, key_len={key_len}"
                )
            padding_mask = attention_mask[:, None, None, :] == 0
            attn_scores = attn_scores.masked_fill(padding_mask, float("-inf"))

        # 6. softmax 后对 V 加权求和。
        attn_weights = F.softmax(attn_scores.float(), dim=-1).to(dtype=query_states.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states)

        # 7. 还原回 [batch, query_len, heads, head_dim]。
        return attn_output.transpose(1, 2).contiguous()


class MiniMindMLP(nn.Module):
    """MiniMind 的前馈网络，也就是 SwiGLU MLP。
    结构是：
    hidden_states -> gate_proj 和 up_proj
    silu(gate_proj) * up_proj
    down_proj 投回 hidden_size
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.hidden_act = config.hidden_act
        if self.hidden_act != "silu":
            raise ValueError(f"当前 MiniMindMLP 只实现 silu，收到 hidden_act={self.hidden_act!r}")
        # gate_proj 和 up_proj 都从 hidden_size 升维到 intermediate_size。
        # down_proj 再从 intermediate_size 投回 hidden_size。
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """执行 SwiGLU 前馈网络。"""

        gate_states = self.gate_proj(hidden_states)
        up_states = self.up_proj(hidden_states)

        # SwiGLU 的核心：一个分支做 silu 门控，另一个分支提供内容。
        hidden_states = F.silu(gate_states) * up_states
        hidden_states = self.down_proj(hidden_states)
        return hidden_states


class MiniMindBlock(nn.Module):
    """一层完整的 MiniMind Transformer Block。
    结构是 Pre-Norm：
    hidden_states -> RMSNorm -> Attention -> 残差连接
    hidden_states -> RMSNorm -> MLP       -> 残差连接
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()

        self.self_attn = MiniMindAttention(config)
        self.mlp = MiniMindMLP(config)

        # Attention 前的归一化。
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MLP 前的归一化。
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: RotaryEmbedding,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """执行一层 Transformer Block。

        hidden_states 形状：[batch, seq_len, hidden_size]。
        返回新的 hidden_states，以及可选的 present_key_value。
        """

        # 1. Attention 子层：先归一化，再注意力，再残差相加。
        residual = hidden_states
        normed_hidden_states = self.input_layernorm(hidden_states)
        attn_output, present_key_value = self.self_attn(
            hidden_states=normed_hidden_states,
            rotary_emb=rotary_emb,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            start_pos=start_pos,
        )
        hidden_states = residual + attn_output

        # 2. MLP 子层：同样先归一化，再 MLP，再残差相加。
        residual = hidden_states
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normed_hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states, present_key_value

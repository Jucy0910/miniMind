from __future__ import annotations

import math

import torch
from torch import nn


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

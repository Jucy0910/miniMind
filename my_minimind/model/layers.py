from __future__ import annotations

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

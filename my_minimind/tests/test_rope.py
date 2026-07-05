from __future__ import annotations

import torch

from my_minimind.model import RotaryEmbedding


def test_rotary_embedding_cache_shape_and_position_zero() -> None:
    # cos/sin 表的形状应该是 [最大位置长度, head_dim]。
    rope = RotaryEmbedding(
        head_dim=8,
        max_position_embeddings=16,
        rope_theta=1_000_000.0,
    )
    cos = rope.cos_cached
    sin = rope.sin_cached

    assert cos.shape == (16, 8)
    assert sin.shape == (16, 8)

    # 位置 0 的旋转角度是 0，所以 cos=1, sin=0。
    assert torch.allclose(cos[0], torch.ones(8))
    assert torch.allclose(sin[0], torch.zeros(8))


def test_rotary_embedding_forward_keeps_shape_dtype_and_norm() -> None:
    # q/k 形状对齐后续 Attention 中的布局：[batch, seq, heads, head_dim]。
    rope = RotaryEmbedding(
        head_dim=8,
        max_position_embeddings=16,
        rope_theta=1_000_000.0,
    )
    query_states = torch.randn(2, 4, 3, 8, dtype=torch.float32)
    key_states = torch.randn(2, 4, 1, 8, dtype=torch.float32)

    query_embed, key_embed = rope(query_states, key_states)

    assert query_embed.shape == query_states.shape
    assert key_embed.shape == key_states.shape
    assert query_embed.dtype == query_states.dtype
    assert key_embed.dtype == key_states.dtype

    # RoPE 是旋转变换，理论上不会改变每个 head 向量的 L2 范数。
    assert torch.allclose(query_embed.norm(dim=-1), query_states.norm(dim=-1), atol=1e-5)
    assert torch.allclose(key_embed.norm(dim=-1), key_states.norm(dim=-1), atol=1e-5)


def test_rotary_embedding_position_zero_unchanged() -> None:
    # 第 0 个位置的 cos=1, sin=0，因此 q/k 应该保持不变。
    rope = RotaryEmbedding(
        head_dim=8,
        max_position_embeddings=16,
        rope_theta=1_000_000.0,
    )
    query_states = torch.randn(1, 1, 2, 8)
    key_states = torch.randn(1, 1, 1, 8)

    query_embed, key_embed = rope(query_states, key_states)

    assert torch.allclose(query_embed, query_states)
    assert torch.allclose(key_embed, key_states)


if __name__ == "__main__":
    test_rotary_embedding_cache_shape_and_position_zero()
    test_rotary_embedding_forward_keeps_shape_dtype_and_norm()
    test_rotary_embedding_position_zero_unchanged()
    print("RoPE 冒烟测试通过")

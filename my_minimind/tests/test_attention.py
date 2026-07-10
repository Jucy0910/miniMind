from __future__ import annotations

import torch

from my_minimind.model import MiniMindAttention, MiniMindConfig, RotaryEmbedding


def build_tiny_attention() -> tuple[MiniMindAttention, RotaryEmbedding]:
    # 小配置用于快速测试，不占太多显存和时间。
    config = MiniMindConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        dropout=0.0,
    )
    attention = MiniMindAttention(config)
    rotary_emb = RotaryEmbedding(
        head_dim=config.head_dim,
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=config.rope_theta,
        rope_scaling=config.rope_scaling,
    )
    return attention, rotary_emb


def test_attention_forward_shape() -> None:
    attention, rotary_emb = build_tiny_attention()
    hidden_states = torch.randn(2, 5, 16)

    output, present_key_value = attention(hidden_states, rotary_emb)

    assert output.shape == hidden_states.shape
    assert present_key_value is None


def test_attention_gqa_projection_shapes() -> None:
    attention, _ = build_tiny_attention()

    # Q 使用 4 个 head，总维度 4 * 4 = 16。
    # K/V 使用 2 个 head，总维度 2 * 4 = 8。
    assert attention.q_proj.out_features == 16
    assert attention.k_proj.out_features == 8
    assert attention.v_proj.out_features == 8
    assert attention.o_proj.in_features == 16
    assert attention.num_key_value_groups == 2


def test_attention_compute_attention_shape() -> None:
    attention, _ = build_tiny_attention()
    query_states = torch.randn(2, 5, 4, 4)
    key_states = torch.randn(2, 5, 2, 4)
    value_states = torch.randn(2, 5, 2, 4)

    output = attention._compute_attention(query_states, key_states, value_states)

    assert output.shape == query_states.shape


def test_attention_cache_shape() -> None:
    attention, rotary_emb = build_tiny_attention()
    hidden_states = torch.randn(2, 5, 16)

    output, present_key_value = attention(hidden_states, rotary_emb, use_cache=True)

    assert output.shape == hidden_states.shape
    assert present_key_value is not None

    key_states, value_states = present_key_value
    assert key_states.shape == (2, 5, 2, 4)
    assert value_states.shape == (2, 5, 2, 4)


def test_attention_cache_append() -> None:
    attention, rotary_emb = build_tiny_attention()

    first_hidden_states = torch.randn(2, 3, 16)
    _, past_key_value = attention(first_hidden_states, rotary_emb, use_cache=True)

    next_hidden_states = torch.randn(2, 2, 16)
    output, present_key_value = attention(
        next_hidden_states,
        rotary_emb,
        past_key_value=past_key_value,
        use_cache=True,
    )

    assert output.shape == next_hidden_states.shape
    assert present_key_value is not None

    key_states, value_states = present_key_value
    assert key_states.shape == (2, 5, 2, 4)
    assert value_states.shape == (2, 5, 2, 4)


if __name__ == "__main__":
    test_attention_forward_shape()
    test_attention_gqa_projection_shapes()
    test_attention_compute_attention_shape()
    test_attention_cache_shape()
    test_attention_cache_append()
    print("Attention 冒烟测试通过")

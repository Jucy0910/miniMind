from __future__ import annotations

import torch

from my_minimind.model import MiniMindBlock, MiniMindConfig, RotaryEmbedding


def build_tiny_block() -> tuple[MiniMindBlock, RotaryEmbedding]:
    # 小配置用于快速验证一层 Transformer Block。
    config = MiniMindConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        dropout=0.0,
    )
    block = MiniMindBlock(config)
    rotary_emb = RotaryEmbedding(
        head_dim=config.head_dim,
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=config.rope_theta,
        rope_scaling=config.rope_scaling,
    )
    return block, rotary_emb


def test_block_forward_shape() -> None:
    block, rotary_emb = build_tiny_block()
    hidden_states = torch.randn(2, 5, 16)

    output, present_key_value = block(hidden_states, rotary_emb)

    assert output.shape == hidden_states.shape
    assert present_key_value is None


def test_block_cache_shape() -> None:
    block, rotary_emb = build_tiny_block()
    hidden_states = torch.randn(2, 5, 16)

    output, present_key_value = block(hidden_states, rotary_emb, use_cache=True)

    assert output.shape == hidden_states.shape
    assert present_key_value is not None

    key_states, value_states = present_key_value
    assert key_states.shape == (2, 5, 2, 4)
    assert value_states.shape == (2, 5, 2, 4)


def test_block_cache_append() -> None:
    block, rotary_emb = build_tiny_block()

    first_hidden_states = torch.randn(2, 3, 16)
    _, past_key_value = block(first_hidden_states, rotary_emb, use_cache=True)

    next_hidden_states = torch.randn(2, 2, 16)
    output, present_key_value = block(
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


def test_block_backward() -> None:
    block, rotary_emb = build_tiny_block()
    hidden_states = torch.randn(2, 5, 16, requires_grad=True)

    output, _ = block(hidden_states, rotary_emb)
    loss = output.sum()
    loss.backward()

    assert hidden_states.grad is not None
    assert block.self_attn.q_proj.weight.grad is not None
    assert block.mlp.gate_proj.weight.grad is not None


if __name__ == "__main__":
    test_block_forward_shape()
    test_block_cache_shape()
    test_block_cache_append()
    test_block_backward()
    print("Block 冒烟测试通过")

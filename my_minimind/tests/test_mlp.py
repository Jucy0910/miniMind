from __future__ import annotations

import torch
import torch.nn.functional as F

from my_minimind.model import MiniMindConfig, MiniMindMLP


def build_tiny_mlp() -> MiniMindMLP:
    config = MiniMindConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        dropout=0.0,
    )
    return MiniMindMLP(config)


def test_mlp_forward_shape() -> None:
    mlp = build_tiny_mlp()
    hidden_states = torch.randn(2, 5, 16)

    output = mlp(hidden_states)

    assert output.shape == hidden_states.shape


def test_mlp_projection_shapes() -> None:
    mlp = build_tiny_mlp()

    assert mlp.gate_proj.in_features == 16
    assert mlp.gate_proj.out_features == 32
    assert mlp.up_proj.in_features == 16
    assert mlp.up_proj.out_features == 32
    assert mlp.down_proj.in_features == 32
    assert mlp.down_proj.out_features == 16


def test_mlp_matches_manual_formula() -> None:
    mlp = build_tiny_mlp()
    hidden_states = torch.randn(2, 5, 16)

    output = mlp(hidden_states)
    expected = mlp.down_proj(F.silu(mlp.gate_proj(hidden_states)) * mlp.up_proj(hidden_states))

    assert torch.allclose(output, expected)


def test_mlp_backward() -> None:
    mlp = build_tiny_mlp()
    hidden_states = torch.randn(2, 5, 16, requires_grad=True)

    loss = mlp(hidden_states).sum()
    loss.backward()

    assert hidden_states.grad is not None
    assert mlp.gate_proj.weight.grad is not None
    assert mlp.up_proj.weight.grad is not None
    assert mlp.down_proj.weight.grad is not None


if __name__ == "__main__":
    test_mlp_forward_shape()
    test_mlp_projection_shapes()
    test_mlp_matches_manual_formula()
    test_mlp_backward()
    print("MLP 冒烟测试通过")

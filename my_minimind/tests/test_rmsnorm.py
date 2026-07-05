from __future__ import annotations

import torch

from my_minimind.model import RMSNorm


def test_rmsnorm_keeps_shape_and_dtype() -> None:
    # RMSNorm 不应该改变输入张量的形状和 dtype。
    layer = RMSNorm(hidden_size=8, eps=1e-6)
    hidden_states = torch.randn(2, 4, 8, dtype=torch.float32)

    output = layer(hidden_states)

    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype


def test_rmsnorm_matches_manual_formula() -> None:
    # 手动按公式算一遍，确认实现没有维度或广播错误。
    layer = RMSNorm(hidden_size=3, eps=1e-6)
    hidden_states = torch.tensor([[[1.0, 2.0, 3.0]]])

    output = layer(hidden_states)
    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    expected = hidden_states * torch.rsqrt(variance + layer.eps)

    assert torch.allclose(output, expected, atol=1e-6)


def test_rmsnorm_has_trainable_weight_and_backward() -> None:
    # weight 必须参与训练；反向传播后应该能得到梯度。
    layer = RMSNorm(hidden_size=8, eps=1e-6)
    hidden_states = torch.randn(2, 4, 8, requires_grad=True)

    output = layer(hidden_states)
    loss = output.sum()
    loss.backward()

    assert layer.weight.requires_grad is True
    assert layer.weight.grad is not None
    assert hidden_states.grad is not None


if __name__ == "__main__":
    test_rmsnorm_keeps_shape_and_dtype()
    test_rmsnorm_matches_manual_formula()
    test_rmsnorm_has_trainable_weight_and_backward()
    print("RMSNorm 冒烟测试通过")

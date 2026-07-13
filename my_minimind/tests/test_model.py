from __future__ import annotations

import torch

from my_minimind.model import MiniMindConfig, MiniMindModel


def build_tiny_model() -> MiniMindModel:
    # 两层小模型足以验证“多层 Block”是否正确串联。
    config = MiniMindConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        dropout=0.0,
    )
    return MiniMindModel(config)


def test_model_forward_shape() -> None:
    model = build_tiny_model()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))

    hidden_states, present_key_values = model(input_ids)

    assert hidden_states.shape == (2, 5, 16)
    assert present_key_values is None


def test_model_has_expected_layers_and_shared_rope() -> None:
    model = build_tiny_model()

    assert len(model.layers) == 2

    # RoPE 只存在于完整模型上，Block/Attention 接收并使用这一份共享组件。
    assert hasattr(model, "rotary_emb")
    for layer in model.layers:
        assert not hasattr(layer, "rotary_emb")
        assert not hasattr(layer.self_attn, "rotary_emb")


def test_model_cache_shape() -> None:
    model = build_tiny_model()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))

    hidden_states, present_key_values = model(input_ids, use_cache=True)

    assert hidden_states.shape == (2, 5, 16)
    assert present_key_values is not None
    assert len(present_key_values) == 2

    for key_states, value_states in present_key_values:
        assert key_states.shape == (2, 5, 2, 4)
        assert value_states.shape == (2, 5, 2, 4)


def test_model_cache_append() -> None:
    model = build_tiny_model()

    first_input_ids = torch.randint(0, model.config.vocab_size, (2, 3))
    _, past_key_values = model(first_input_ids, use_cache=True)

    next_input_ids = torch.randint(0, model.config.vocab_size, (2, 2))
    hidden_states, present_key_values = model(
        next_input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )

    assert hidden_states.shape == (2, 2, 16)
    assert present_key_values is not None

    for key_states, value_states in present_key_values:
        assert key_states.shape == (2, 5, 2, 4)
        assert value_states.shape == (2, 5, 2, 4)


def test_model_backward() -> None:
    model = build_tiny_model()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))

    hidden_states, _ = model(input_ids)
    loss = hidden_states.pow(2).mean()
    loss.backward()

    assert model.embed_tokens.weight.grad is not None
    assert model.layers[0].self_attn.q_proj.weight.grad is not None
    assert model.layers[1].mlp.gate_proj.weight.grad is not None


if __name__ == "__main__":
    test_model_forward_shape()
    test_model_has_expected_layers_and_shared_rope()
    test_model_cache_shape()
    test_model_cache_append()
    test_model_backward()
    print("MiniMindModel 冒烟测试通过")

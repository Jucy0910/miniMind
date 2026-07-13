from __future__ import annotations

import torch
from torch.nn import functional as F

from my_minimind.model import MiniMindConfig, MiniMindForCausalLM


def build_tiny_causal_lm() -> MiniMindForCausalLM:
    """创建一个很小的因果语言模型，降低测试的计算量。"""

    config = MiniMindConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        dropout=0.0,
        tie_word_embeddings=True,
    )
    return MiniMindForCausalLM(config)


def test_logits_shape_without_labels() -> None:
    """不传 labels 时应该只计算词表分数，不计算 loss。"""

    model = build_tiny_causal_lm()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))

    output = model(input_ids=input_ids)

    assert output.logits.shape == (2, 5, 32)
    assert output.loss is None
    assert output.past_key_values is None


def test_causal_lm_loss_uses_next_token_shift() -> None:
    """验证模型内部确实使用错开一位后的 logits 和 labels 计算 loss。"""

    model = build_tiny_causal_lm()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    labels = input_ids.clone()

    # -100 表示这个目标位置不参与监督。
    labels[0, 3] = -100

    output = model(input_ids=input_ids, labels=labels)

    assert output.loss is not None
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)

    # 在测试中手动执行一次同样的 next-token 错位和交叉熵计算。
    expected_logits = output.logits[:, :-1, :].contiguous().view(-1, 32)
    expected_labels = labels[:, 1:].contiguous().view(-1)
    expected_loss = F.cross_entropy(
        expected_logits,
        expected_labels,
        ignore_index=-100,
    )

    assert torch.allclose(output.loss, expected_loss)


def test_embedding_and_lm_head_share_weight() -> None:
    """开启权重共享后，Embedding 和 LM Head 必须指向同一块参数内存。"""

    model = build_tiny_causal_lm()

    embedding_weight_address = model.model.embed_tokens.weight.data_ptr()
    lm_head_weight_address = model.lm_head.weight.data_ptr()

    assert embedding_weight_address == lm_head_weight_address


def test_loss_can_backward() -> None:
    """loss 应该能够反向传播到 Embedding、Attention 和 MLP。"""

    model = build_tiny_causal_lm()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))

    output = model(input_ids=input_ids, labels=input_ids.clone())
    assert output.loss is not None
    output.loss.backward()

    assert model.model.embed_tokens.weight.grad is not None
    assert model.model.layers[0].self_attn.q_proj.weight.grad is not None
    assert model.model.layers[1].mlp.gate_proj.weight.grad is not None


def test_causal_lm_returns_cache() -> None:
    """推理开启 cache 时，完整模型应原样返回主干生成的逐层 KV cache。"""

    model = build_tiny_causal_lm()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))

    output = model(input_ids=input_ids, use_cache=True)

    assert output.past_key_values is not None
    assert len(output.past_key_values) == model.config.num_hidden_layers


if __name__ == "__main__":
    test_logits_shape_without_labels()
    test_causal_lm_loss_uses_next_token_shift()
    test_embedding_and_lm_head_share_weight()
    test_loss_can_backward()
    test_causal_lm_returns_cache()
    print("MiniMindForCausalLM 冒烟测试通过")

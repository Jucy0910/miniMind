from __future__ import annotations

from my_minimind.model import MiniMindConfig


def test_default_config_aligns_minimind_dense() -> None:
    # 默认配置应该对齐原始 MiniMind dense 模型的核心结构参数。
    config = MiniMindConfig()

    assert config.vocab_size == 6400
    assert config.hidden_size == 768
    assert config.num_hidden_layers == 8
    assert config.num_attention_heads == 8
    assert config.num_key_value_heads == 4
    assert config.head_dim == 96
    assert config.intermediate_size == 2432
    assert config.max_position_embeddings == 32768
    assert config.rope_theta == 1_000_000.0
    assert config.rms_norm_eps == 1e-6
    assert config.hidden_act == "silu"
    assert config.tie_word_embeddings is True


def test_config_from_dict_ignores_unrelated_fields() -> None:
    # 原始 config.json 里有一些 Hugging Face 相关字段。
    # 我们只吸收手写模型真正需要的字段，其余字段自动忽略。
    config = MiniMindConfig.from_dict(
        {
            "architectures": ["Qwen3ForCausalLM"],
            "transformers_version": "4.57.6",
            "hidden_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        }
    )

    assert config.hidden_size == 128
    assert config.num_attention_heads == 4
    assert config.num_key_value_heads == 2
    assert config.head_dim == 32


if __name__ == "__main__":
    test_default_config_aligns_minimind_dense()
    test_config_from_dict_ignores_unrelated_fields()
    print("模型配置冒烟测试通过")

import torch

from my_minimind.configs import MiniMindConfig
from my_minimind.model import MiniMindForCausalLM


def test_stage1_model_forward_shape():
    # 构造一个极小模型，避免冒烟测试占用太多显存或时间。
    # vocab_size=32 表示词表只有 32 个 token。
    # hidden_size=16 表示每个 token 只映射成 16 维向量。
    config = MiniMindConfig(vocab_size=32, hidden_size=16, num_attention_heads=4)
    model = MiniMindForCausalLM(config)

    # 随机生成一批 token id。
    # 形状 [2, 8] 表示 batch_size=2, seq_len=8。
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    # 第一阶段直接让 labels 等于 input_ids。
    # 模型内部会自动错位，使用第 t 个位置预测第 t+1 个 token。
    labels = input_ids.clone()

    output = model(input_ids, labels=labels)

    # logits 的形状必须是 [batch_size, seq_len, vocab_size]。
    assert output.logits.shape == (2, 8, 32)

    # 传入 labels 后必须能算出 loss，说明 forward 和 loss 对齐没有明显错误。
    assert output.loss is not None


if __name__ == "__main__":
    # 这样写可以让该文件既能被 pytest 收集，
    # 也能直接用 python -m my_minimind.tests.test_stage1_smoke 运行。
    test_stage1_model_forward_shape()
    print("第一阶段冒烟测试通过")

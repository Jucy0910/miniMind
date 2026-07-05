from dataclasses import dataclass

import torch
from torch import nn

from my_minimind.configs import MiniMindConfig


@dataclass
class CausalLMOutput:
    # logits 是模型对每个位置、每个词表 token 的打分。
    # 形状是 [batch_size, seq_len, vocab_size]。
    logits: torch.Tensor

    # 如果 forward 时传入 labels，就返回语言模型 loss。
    # 如果只做生成或评测，也可以不传 labels，此时 loss 为 None。
    loss: torch.Tensor | None = None


class MiniMindForCausalLM(nn.Module):
    """第一阶段的模型外壳。

    Transformer 内部结构留到第二阶段实现。现在先固定公开的 forward 接口，
    让数据集、训练、checkpoint 和评测代码在模型变复杂前就使用同一套约定。
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()

        # 保存配置，后续保存 checkpoint 或生成时会用到。
        self.config = config

        # token embedding：把离散 token id 映射为连续向量。
        # 输入形状 [batch, seq_len]，输出形状 [batch, seq_len, hidden_size]。
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # lm_head：把 hidden state 投影回词表空间。
        # 输出 logits 形状 [batch, seq_len, vocab_size]。
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 权重共享：输入 embedding 和输出 lm_head 使用同一个权重矩阵。
        # 这在 GPT 类模型中很常见，既减少参数，也常常有利于训练。
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> CausalLMOutput:
        # 第一阶段还没有 TransformerBlock。
        # 当前 forward 只是验证完整 CausalLM 接口：
        # input_ids -> embedding -> logits -> optional loss。
        hidden_states = self.embed_tokens(input_ids)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Causal LM 的训练目标是“用当前位置之前的信息预测下一个 token”。
            # 因此第 t 个 logits 预测第 t+1 个 label：
            #   logits[:, :-1, :] 对齐 labels[:, 1:]
            loss = nn.functional.cross_entropy(
                # cross_entropy 需要形状 [N, vocab_size] 的 logits。
                # 所以这里把 batch 和 seq 维度合并。
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                # labels 也展平成 [N]。
                # 其中 -100 的位置会被 ignore_index 忽略。
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(logits=logits, loss=loss)

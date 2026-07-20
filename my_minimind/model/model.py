from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import MiniMindConfig
from .layers import MiniMindBlock, RMSNorm, RotaryEmbedding


# 单层 KV cache：第一个张量是 Key，第二个张量是 Value。
# 两者形状都是 [batch, cache_len, num_key_value_heads, head_dim]。
KVCache = tuple[torch.Tensor, torch.Tensor]

# 完整模型有多层 Transformer，所以每一层都需要保存自己的 KV cache。
PastKeyValues = tuple[KVCache, ...]


@dataclass
class CausalLMOutput:
    """因果语言模型一次前向传播的输出。
    使用具名字段保存结果，比依靠元组中第几个位置更容易阅读：
    - logits：模型对每个位置、每个词表 token 给出的原始分数；
    - loss：传入 labels 时计算得到的语言模型损失，否则为 None；
    - past_key_values：use_cache=True 时返回的逐层 KV cache，否则为 None。
    """

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: PastKeyValues | None = None


class MiniMindModel(nn.Module):
    """MiniMind 基础模型主干。
    这个类负责：
    input_ids -> Token Embedding -> 多层 MiniMindBlock -> Final RMSNorm。
    它输出 hidden_states，不负责投影到词表，也不负责计算语言模型 loss。
    这些工作由文件下方的 MiniMindForCausalLM 负责。
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        # 1. Token Embedding：把离散 token id 转换为 hidden_size 维向量。
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.dropout)
        # 2. 整个模型只创建一份 RoPE cos/sin 缓存。
        #    每一层 Attention 都使用同一个 RotaryEmbedding，避免重复占用内存。
        self.rotary_emb = RotaryEmbedding(
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        # 3. 按配置创建多层 Transformer Block。
        self.layers = nn.ModuleList(
            [MiniMindBlock(config) for _ in range(config.num_hidden_layers)]
        )
        # 4. 所有 Transformer Block 结束后，再做一次最终 RMSNorm。
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 5. 按 MiniMindConfig.initializer_range 初始化可训练权重。
        self.apply(self._init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, PastKeyValues | None]:
        """执行完整 MiniMind 主干前向传播。
        input_ids 形状：[batch, seq_len]。
        返回 hidden_states 形状：[batch, seq_len, hidden_size]。
        use_cache=True 时，第二个返回值包含每一层的新 KV cache。
        """
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids 必须是二维张量 [batch, seq_len]，"
                f"当前形状为 {tuple(input_ids.shape)}"
            )
        # 没有历史 cache 时，为每一层准备一个 None 占位。
        if past_key_values is None:
            layer_past_key_values: tuple[KVCache | None, ...] = tuple(
                None for _ in range(self.num_hidden_layers)
            )
            start_pos = 0
        else:
            if len(past_key_values) != self.num_hidden_layers:
                raise ValueError(
                    "past_key_values 的层数必须等于 num_hidden_layers，"
                    f"当前 cache 层数={len(past_key_values)}, "
                    f"模型层数={self.num_hidden_layers}"
                )
            layer_past_key_values = past_key_values
            # 所有层 cache 的序列长度应该一致，取第一层长度作为当前 RoPE 起始位置。
            start_pos = past_key_values[0][0].shape[1]

        # 1. token id -> token embedding。
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.embed_dropout(hidden_states)

        # use_cache=True 时，把每层新产生的 KV cache 收集起来。
        present_key_values: list[KVCache] = []

        # 2. hidden_states 依次通过每一个 Transformer Block。
        for layer, past_key_value in zip(self.layers, layer_past_key_values):
            hidden_states, present_key_value = layer(
                hidden_states=hidden_states,
                rotary_emb=self.rotary_emb,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                start_pos=start_pos,
            )

            if use_cache:
                # use_cache=True 时，Block 一定会返回当前层的新 cache。
                if present_key_value is None:
                    raise RuntimeError("use_cache=True，但 Transformer Block 没有返回 KV cache")
                present_key_values.append(present_key_value)

        # 3. 多层 Block 结束后做最终归一化。
        hidden_states = self.norm(hidden_states)

        if use_cache:
            return hidden_states, tuple(present_key_values)
        return hidden_states, None

    def _init_weights(self, module: nn.Module) -> None:
        """初始化 Linear 和 Embedding 的可训练权重。"""

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


class MiniMindForCausalLM(nn.Module):
    """用于因果语言建模的完整 MiniMind 模型。
    MiniMindModel 只负责把 input_ids 转换成 hidden_states。
    这个类在模型主干外再增加两部分：
    1. LM Head：把 hidden_size 维的隐藏状态投影成 vocab_size 维词表分数；
    2. Causal LM loss：让当前位置预测它后面的下一个 token。
    因此，预训练时可以直接调用：
        output = model(input_ids=input_ids, labels=labels)
        loss = output.loss
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config

        # 1. MiniMindModel 是已经完成的 Transformer 主干。
        self.model = MiniMindModel(config)

        # 2. LM Head 把每个 token 的隐藏向量映射为整个词表上的预测分数。
        #    输入形状：[batch, seq_len, hidden_size]
        #    输出形状：[batch, seq_len, vocab_size]
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        if config.tie_word_embeddings:
            # 输入 Embedding 和输出 LM Head 共享同一块权重。
            # 两者虽然用途不同，但权重形状都是 [vocab_size, hidden_size]。
            # 共享后可以减少参数量，并让同一个 token 的输入、输出表示相互约束。
            self.lm_head.weight = self.model.embed_tokens.weight #模型内部会自行进行转置
        else:
            # 不共享权重时，LM Head 是一份独立参数，需要单独初始化。
            # 这里不调用 self.apply，避免把主干中已经初始化的参数再次初始化。
            nn.init.normal_(
                self.lm_head.weight,
                mean=0.0,
                std=config.initializer_range,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        """执行因果语言模型前向传播，并按需计算 next-token loss。

        input_ids 和 labels 的形状都应该是 [batch, seq_len]。
        labels 中等于 -100 的位置会被交叉熵忽略，不参与 loss 计算。
        """

        # 1. Transformer 主干产生每个 token 对应的隐藏状态。
        hidden_states, present_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # 2. LM Head 为序列中每个位置生成完整词表上的原始预测分数。
        #    logits 还没有经过 softmax，因为 CrossEntropyLoss 内部会完成该计算。
        logits = self.lm_head(hidden_states)

        # 推理时通常不传 labels，此时只需要返回 logits 和可选的 KV cache。
        loss: torch.Tensor | None = None

        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError(
                    "labels 的形状必须和 input_ids 完全一致，"
                    f"当前 labels.shape={tuple(labels.shape)}, "
                    f"input_ids.shape={tuple(input_ids.shape)}"
                )

            if input_ids.shape[1] < 2:
                raise ValueError("计算语言模型 loss 时，序列长度至少需要为 2")

            # 3. 因果语言模型学习“根据前文预测下一个 token”。
            #
            # 假设原始序列为：[我, 爱, 学, 习]
            # 输入位置使用：  [我, 爱, 学]
            # 监督目标使用：  [爱, 学, 习]
            #
            # 所以 logits 去掉最后一个位置，labels 去掉第一个位置，
            # 二者便能在时间维度上一一对应。
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # 4. CrossEntropy 接收二维预测分数和一维目标 token id，
            #    因此把 batch 和 seq_len 两个维度合并到一起。
            flat_logits = shift_logits.view(-1, self.config.vocab_size)
            flat_labels = shift_labels.view(-1)

            # ignore_index=-100 与 Dataset 产生的 labels 约定一致。
            # padding、问题部分等不需要监督的位置会被完全排除在 loss 之外。
            loss = F.cross_entropy(
                flat_logits,
                flat_labels,
                ignore_index=-100,
            )
        return CausalLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=present_key_values,
        )

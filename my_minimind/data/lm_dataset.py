from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


# PyTorch 的 CrossEntropyLoss 默认会忽略 label=-100 的位置。
# 训练语言模型时，padding、用户提问、system 提示等不该算 loss 的位置都用它屏蔽。
IGNORE_INDEX = -100


@dataclass(frozen=True)
class LMSample:
    """语言模型训练样本。
    input_ids 是喂给模型的 token 序列。
    labels 是监督目标，和 input_ids 等长；不参与 loss 的位置填 IGNORE_INDEX。
    """
    input_ids: torch.Tensor
    labels: torch.Tensor


def pre_processing_chat(
    conversations: list[dict[str, Any]],
    add_system_ratio: float = 0.2,
) -> list[dict[str, Any]]:
    """对 SFT 对话做和原始 MiniMind 相同的轻量预处理。
    原始项目会以一定概率给没有 system 的普通对话补一个 system prompt。
    tool-use 数据结构更复杂，直接保留，避免破坏工具调用格式。
    """
    if any(message.get("tools") for message in conversations):
        return conversations

    system_prompts = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model.",
    ]

    if conversations and conversations[0].get("role") != "system":
        if random.random() < add_system_ratio:
            system_message = {
                "role": "system",
                "content": random.choice(system_prompts),
            }
            return [system_message] + conversations
    return conversations


def post_processing_chat(prompt: str, empty_think_ratio: float = 0.2) -> str:
    """对 chat_template 生成后的文本做和原始 MiniMind 相同的后处理。"""

    # tokenizer_config 里的模板会给 assistant 消息添加空 thinking 块。
    # 原始项目按概率保留一部分空 thinking，其余样本移除，让训练数据更多样。
    empty_think = "<think>\n\n</think>\n\n"
    if empty_think in prompt and random.random() > empty_think_ratio:
        prompt = prompt.replace(empty_think, "")
    return prompt


class JsonlDataset(Dataset):
    """基于 JSONL 行偏移的数据集基类。
    这里只保存每一行在文件里的 byte offset，不保存完整样本内容。
    好处是大文件训练时内存占用稳定；真正取样本时再 seek 到对应行读取。
    """
    def __init__(self, data_path: str | Path):
        super().__init__()
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"找不到数据集文件：{self.data_path}")
        self.offsets = self._build_offsets(self.data_path)
        if not self.offsets:
            raise ValueError(f"数据集文件为空：{self.data_path}")
        # 多进程 DataLoader 会 pickle Dataset。
        # 文件句柄不能安全共享，所以这里延迟到每个 worker 内部再打开。
        self._file = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self) -> dict[str, Any]:
        """让 DataLoader worker 拿到 Dataset 时重新打开自己的文件句柄。"""

        state = self.__dict__.copy()
        state["_file"] = None
        return state

    @staticmethod
    def _build_offsets(path: Path) -> list[int]:
        """扫描 JSONL 文件，记录每一条非空样本的起始位置。"""

        offsets: list[int] = []
        with path.open("rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
        return offsets

    def _get_file(self):
        """获取当前进程自己的文件句柄。"""

        if self._file is None or self._file.closed:
            self._file = self.data_path.open("rb")
        return self._file

    def read_record(self, index: int) -> dict[str, Any]:
        """读取指定下标的 JSON 对象。"""

        f = self._get_file()
        f.seek(self.offsets[index])
        line = f.readline().decode("utf-8", errors="ignore")
        return json.loads(line)


class PretrainDataset(JsonlDataset):
    """Causal LM 预训练数据集。
    期望 JSONL 每行至少包含：
    {"text": "..."}
    """
    def __init__(
        self,
        data_path: str | Path,
        tokenizer: Any,
        max_length: int = 512,
        text_field: str = "text",
    ):
        super().__init__(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.read_record(index)
        if self.text_field not in record:
            raise ValueError(f"样本缺少字段：{self.text_field}")

        # 预训练只学习连续文本本身，不走 chat_template。
        text = str(record[self.text_field])
        token_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids

        # 显式拼接 BOS/EOS，和原始 MiniMind 的预训练数据处理一致。
        token_ids = [self.tokenizer.bos_token_id] + token_ids + [self.tokenizer.eos_token_id]
        input_ids = self._pad_to_max_length(token_ids)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = IGNORE_INDEX
        return input_ids, labels

    def _pad_to_max_length(self, token_ids: list[int]) -> torch.Tensor:
        """把一条样本补齐到固定长度，方便 DataLoader 堆叠 batch。"""

        pad_count = self.max_length - len(token_ids)
        if pad_count < 0:
            token_ids = token_ids[: self.max_length]
            pad_count = 0
        token_ids = token_ids + [self.tokenizer.pad_token_id] * pad_count
        return torch.tensor(token_ids, dtype=torch.long)


class SFTDataset(JsonlDataset):
    """监督微调 SFT 数据集。

    期望 JSONL 每行至少包含：
    {"conversations": [{"role": "...", "content": "..."}]}

    SFT 阶段只对 assistant 回复计算 loss，system/user/tool 等上下文只作为输入条件。
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: Any,
        max_length: int = 1024,
        conversation_field: str = "conversations",
    ):
        super().__init__(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conversation_field = conversation_field

        # 原始 MiniMind 通过查找 "<|im_start|>assistant\n" 和 "<|im_end|>\n"
        # 来定位 assistant 回复区间，我们这里保持同一套规则。
        self.assistant_start_ids = tokenizer(
            f"{tokenizer.bos_token}assistant\n",
            add_special_tokens=False,
        ).input_ids
        self.message_end_ids = tokenizer(
            f"{tokenizer.eos_token}\n",
            add_special_tokens=False,
        ).input_ids

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.read_record(index)
        if self.conversation_field not in record:
            raise ValueError(f"样本缺少字段：{self.conversation_field}")

        conversations = pre_processing_chat(record[self.conversation_field])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)

        # SFT prompt 已经由 chat_template 写入结构 token，所以这里直接截断即可。
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids = self._pad_ids(input_ids)
        labels = self.generate_labels(input_ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def create_chat_prompt(self, conversations: list[dict[str, Any]]) -> str:
        """把多轮对话转成模型真正看到的一整段训练文本。"""

        messages: list[dict[str, Any]] = []
        tools = None
        for message in conversations:
            # 复制一份，避免修改原始样本对象。
            message = dict(message)

            # tool 定义通常挂在 system 消息上；有些 JSONL 会把它保存成字符串。
            if message.get("role") == "system" and message.get("tools"):
                raw_tools = message["tools"]
                tools = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools

            # tool_calls 也可能被保存成 JSON 字符串，模板需要的是结构化对象。
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
        )

    def generate_labels(self, input_ids: list[int]) -> list[int]:
        """生成 SFT labels，只保留 assistant 回复区间的监督信号。"""

        labels = [IGNORE_INDEX] * len(input_ids)
        cursor = 0
        while cursor < len(input_ids):
            start_match = input_ids[cursor : cursor + len(self.assistant_start_ids)]
            if start_match != self.assistant_start_ids:
                cursor += 1
                continue

            # assistant_start_ids 后面才是真正要学习的 assistant 内容。
            start = cursor + len(self.assistant_start_ids)
            end = start
            while end < len(input_ids):
                end_match = input_ids[end : end + len(self.message_end_ids)]
                if end_match == self.message_end_ids:
                    break
                end += 1

            # end token 本身也让模型学习，因为生成时需要知道何时结束本轮消息。
            label_end = min(end + len(self.message_end_ids), len(input_ids), self.max_length)
            for label_index in range(start, label_end):
                if input_ids[label_index] != self.tokenizer.pad_token_id:
                    labels[label_index] = input_ids[label_index]

            cursor = label_end
        return labels

    def _pad_ids(self, input_ids: list[int]) -> list[int]:
        """把 token id 列表补齐到 max_length。"""

        pad_count = self.max_length - len(input_ids)
        if pad_count > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_count
        return input_ids

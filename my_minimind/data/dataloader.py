from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader, Dataset

from .lm_dataset import PretrainDataset, SFTDataset
from .tokenizer import load_tokenizer


# 默认使用我们自己训练出来的 tokenizer。
# 后续如果要临时对齐原始 MiniMind，也可以把这里改成 original_project/model。
DEFAULT_TOKENIZER_PATH = "my_minimind/tokenizer"


def language_model_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """把 Dataset 产出的单条样本合并成一个 batch。

    Dataset 返回的是 (input_ids, labels)。
    训练循环通常更喜欢字典形式，后面可以直接传给 model(**batch) 或手动取字段。
    """

    input_ids, labels = zip(*batch)
    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "labels": torch.stack(labels, dim=0),
    }


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """从 Dataset 构造 PyTorch DataLoader。"""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=language_model_collate_fn,
    )


def create_pretrain_dataloader(
    data_path: str | Path,
    batch_size: int = 16,
    max_length: int = 512,
    tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    tokenizer_kwargs: dict[str, Any] | None = None,
) -> DataLoader:
    """创建预训练 DataLoader。"""

    tokenizer = load_tokenizer(tokenizer_path, **(tokenizer_kwargs or {}))
    dataset = PretrainDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def create_sft_dataloader(
    data_path: str | Path,
    batch_size: int = 8,
    max_length: int = 1024,
    tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    tokenizer_kwargs: dict[str, Any] | None = None,
) -> DataLoader:
    """创建 SFT DataLoader。"""

    tokenizer = load_tokenizer(tokenizer_path, **(tokenizer_kwargs or {}))
    dataset = SFTDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def create_dataloader(
    stage: Literal["pretrain", "sft"],
    data_path: str | Path,
    batch_size: int,
    max_length: int,
    tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """按训练阶段创建 DataLoader。

    stage="pretrain"：读取 text 字段，训练普通续写。
    stage="sft"：读取 conversations 字段，只监督 assistant 回复。
    """

    if stage == "pretrain":
        return create_pretrain_dataloader(
            data_path=data_path,
            batch_size=batch_size,
            max_length=max_length,
            tokenizer_path=tokenizer_path,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
    if stage == "sft":
        return create_sft_dataloader(
            data_path=data_path,
            batch_size=batch_size,
            max_length=max_length,
            tokenizer_path=tokenizer_path,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
    raise ValueError(f"不支持的训练阶段：{stage}")

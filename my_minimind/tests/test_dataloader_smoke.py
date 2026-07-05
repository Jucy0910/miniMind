from __future__ import annotations

import torch

from my_minimind.data import create_pretrain_dataloader, create_sft_dataloader


# 冒烟测试使用前面从大数据集中抽出的 50 条 sample 文件。
# 目的不是验证训练效果，而是快速确认：
# 1. tokenizer 能加载；
# 2. JSONL 能读取；
# 3. DataLoader 能组 batch；
# 4. input_ids/labels 形状和 dtype 正常。
PRETRAIN_SAMPLE_PATH = "datasets/pretrain_t2t_mini_sample.jsonl"
SFT_SAMPLE_PATH = "datasets/sft_t2t_mini_sample.jsonl"


def test_pretrain_dataloader_smoke() -> None:
    dataloader = create_pretrain_dataloader(
        data_path=PRETRAIN_SAMPLE_PATH,
        batch_size=2,
        max_length=64,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    batch = next(iter(dataloader))

    assert batch["input_ids"].shape == (2, 64)
    assert batch["labels"].shape == (2, 64)
    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long


def test_sft_dataloader_smoke() -> None:
    dataloader = create_sft_dataloader(
        data_path=SFT_SAMPLE_PATH,
        batch_size=2,
        max_length=128,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    batch = next(iter(dataloader))

    assert batch["input_ids"].shape == (2, 128)
    assert batch["labels"].shape == (2, 128)
    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long

    # SFT 只监督 assistant 区间，所以 labels 里应该同时存在忽略位置和有效 token。
    assert (batch["labels"] == -100).any()
    assert (batch["labels"] != -100).any()


if __name__ == "__main__":
    test_pretrain_dataloader_smoke()
    test_sft_dataloader_smoke()
    print("DataLoader 冒烟测试通过")

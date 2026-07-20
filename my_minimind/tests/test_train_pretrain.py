from __future__ import annotations

import json
from pathlib import Path
import tempfile

import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset

from my_minimind.model import MiniMindConfig, MiniMindForCausalLM
from my_minimind.train import train_pretrain


def test_learning_rate_schedule() -> None:
    """学习率应该先线性升高，再逐步下降到设定的最小值附近。"""

    first_learning_rate = train_pretrain.calculate_learning_rate(
        update_step=0,
        total_update_steps=100,
        warmup_steps=10,
        max_learning_rate=1e-3,
        min_learning_rate_ratio=0.1,
    )
    warmup_end_learning_rate = train_pretrain.calculate_learning_rate(
        update_step=9,
        total_update_steps=100,
        warmup_steps=10,
        max_learning_rate=1e-3,
        min_learning_rate_ratio=0.1,
    )
    final_learning_rate = train_pretrain.calculate_learning_rate(
        update_step=99,
        total_update_steps=100,
        warmup_steps=10,
        max_learning_rate=1e-3,
        min_learning_rate_ratio=0.1,
    )

    assert first_learning_rate == 1e-4
    assert warmup_end_learning_rate == 1e-3
    assert abs(final_learning_rate - 1e-4) < 1e-12


def test_one_optimizer_update_and_checkpoint() -> None:
    """用极小模型验证前向、反向、参数更新和 checkpoint 能完整串联。"""

    config = MiniMindConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        max_position_embeddings=16,
        dropout=0.0,
    )
    model = MiniMindForCausalLM(config)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    input_ids = torch.randint(0, config.vocab_size, (4, 6))
    labels = input_ids.clone()
    dataset = TensorDataset(input_ids, labels)

    # 测试期间临时换成很小的 batch 和一次训练步，避免启动正式训练配置。
    original_values = {
        "BATCH_SIZE": train_pretrain.BATCH_SIZE,
        "NUM_WORKERS": train_pretrain.NUM_WORKERS,
        "PIN_MEMORY": train_pretrain.PIN_MEMORY,
        "DROP_LAST": train_pretrain.DROP_LAST,
        "GRADIENT_ACCUMULATION_STEPS": train_pretrain.GRADIENT_ACCUMULATION_STEPS,
        "LOG_INTERVAL": train_pretrain.LOG_INTERVAL,
        "SAVE_INTERVAL": train_pretrain.SAVE_INTERVAL,
        "MAX_TRAIN_STEPS": train_pretrain.MAX_TRAIN_STEPS,
        "OUTPUT_DIR": train_pretrain.OUTPUT_DIR,
        "TRAIN_LOG_PATH": train_pretrain.TRAIN_LOG_PATH,
        "METRICS_LOG_PATH": train_pretrain.METRICS_LOG_PATH,
    }

    with tempfile.TemporaryDirectory() as temporary_directory:
        train_pretrain.BATCH_SIZE = 2
        train_pretrain.NUM_WORKERS = 0
        train_pretrain.PIN_MEMORY = False
        train_pretrain.DROP_LAST = True
        train_pretrain.GRADIENT_ACCUMULATION_STEPS = 2
        train_pretrain.LOG_INTERVAL = 100
        train_pretrain.SAVE_INTERVAL = 100
        train_pretrain.MAX_TRAIN_STEPS = 1
        train_pretrain.OUTPUT_DIR = Path(temporary_directory)
        train_pretrain.TRAIN_LOG_PATH = (
            train_pretrain.OUTPUT_DIR / "train.log"
        )
        train_pretrain.METRICS_LOG_PATH = (
            train_pretrain.OUTPUT_DIR / "metrics.jsonl"
        )
        train_pretrain.initialize_log_files(resume_from_checkpoint=False)

        embedding_before_training = model.model.embed_tokens.weight.detach().clone()

        global_update_step, stopped_early = train_pretrain.train_one_epoch(
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            scaler=scaler,
            device=torch.device("cpu"),
            autocast_dtype=None,
            model_config=config,
            epoch=0,
            start_batch_index=0,
            global_update_step=0,
            total_update_steps=1,
            warmup_steps=0,
        )

        assert global_update_step == 1
        assert stopped_early
        assert not torch.equal(
            embedding_before_training,
            model.model.embed_tokens.weight,
        )
        checkpoint_path = Path(temporary_directory) / "latest_checkpoint.pt"
        assert checkpoint_path.exists()

        # 创建一套全新的模型和优化器，确认 checkpoint 不只是能写，还能完整恢复。
        restored_model = MiniMindForCausalLM(config)
        restored_optimizer = AdamW(restored_model.parameters(), lr=1e-3)
        restored_scaler = torch.amp.GradScaler("cpu", enabled=False)
        restored_epoch, restored_batch, restored_step = train_pretrain.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=restored_model,
            optimizer=restored_optimizer,
            scaler=restored_scaler,
            device=torch.device("cpu"),
        )

        assert restored_epoch == 0
        assert restored_batch == 2
        assert restored_step == 1
        assert torch.equal(
            restored_model.model.embed_tokens.weight,
            model.model.embed_tokens.weight,
        )

        # 同一次训练日志应该同时出现在文本文件和结构化指标文件中。
        train_log_text = train_pretrain.TRAIN_LOG_PATH.read_text(encoding="utf-8")
        assert "loss=" in train_log_text
        assert "checkpoint 已保存" in train_log_text

        metrics_lines = train_pretrain.METRICS_LOG_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        assert len(metrics_lines) == 1

        metrics = json.loads(metrics_lines[0])
        assert metrics["update_step"] == 1
        assert "loss" in metrics
        assert "learning_rate" in metrics
        assert "grad_norm" in metrics

    # 恢复模块级参数，避免这个测试影响同一进程中的其他测试。
    for name, value in original_values.items():
        setattr(train_pretrain, name, value)


if __name__ == "__main__":
    test_learning_rate_schedule()
    test_one_optimizer_update_and_checkpoint()
    print("预训练逻辑冒烟测试通过")

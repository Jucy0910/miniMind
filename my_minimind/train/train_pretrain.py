from __future__ import annotations

import os
from pathlib import Path
import sys


# ============================================================================
# 一、训练参数
# ============================================================================
# 所有需要经常调整的参数都集中放在文件顶部，修改后直接运行本文件即可。

# 指定程序可以看到的物理显卡编号。
# 设置 CUDA_VISIBLE_DEVICES 后，下面选择的物理显卡会在程序内部显示为 cuda:0。
GPU_ID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# 把项目根目录加入 Python 模块搜索路径。
# 这样既可以使用 python -m my_minimind.train.train_pretrain，
# 也可以直接使用 python /完整路径/train_pretrain.py。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contextlib import nullcontext
from datetime import datetime
import json
import math
import random
import time
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, Subset

from my_minimind.data import PretrainDataset, build_dataloader, load_tokenizer
from my_minimind.model import MiniMindConfig, MiniMindForCausalLM


# 数据、tokenizer 和训练输出路径。
DATA_PATH = PROJECT_ROOT / "datasets/pretrain_t2t_mini.jsonl"
TOKENIZER_PATH = PROJECT_ROOT / "my_minimind/tokenizer"
OUTPUT_DIR = PROJECT_ROOT / "my_minimind_out/pretrain"

# 完整文本日志：保存与控制台相同的训练信息。
TRAIN_LOG_PATH = OUTPUT_DIR / "train.log"

# 结构化指标日志：每行是一个 JSON 对象，方便训练后绘制 loss 等曲线。
METRICS_LOG_PATH = OUTPUT_DIR / "metrics.jsonl"

# 如果填写 JSON 路径，就从 JSON 加载模型结构；保持 None 则使用 MiniMindConfig 默认值。
MODEL_CONFIG_PATH: Path | None = None

# 基础训练参数。
EPOCHS = 2
BATCH_SIZE = 4
MAX_SEQ_LEN = 340
LEARNING_RATE = 5e-4
MIN_LEARNING_RATE_RATIO = 0.1
WEIGHT_DECAY = 0.01

# 梯度累积 8 次后才更新一次参数。
# 当前有效 batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS = 32。
GRADIENT_ACCUMULATION_STEPS = 8
MAX_GRAD_NORM = 1.0

# 前 WARMUP_RATIO 比例的参数更新步用于线性升温，之后使用余弦下降。
WARMUP_RATIO = 0.03

# 可选值："bfloat16"、"float16"、"float32"。
# bfloat16 通常比 float16 更不容易数值溢出，但需要显卡支持。
MIXED_PRECISION = "bfloat16"

# DataLoader 参数。
NUM_WORKERS = 4
PIN_MEMORY = True
DROP_LAST = True

# 日志和 checkpoint 都以“optimizer 完成一次参数更新”为计数单位。
LOG_INTERVAL = 10
SAVE_INTERVAL = 500

# True：自动读取 OUTPUT_DIR/latest_checkpoint.pt 继续训练。
# False：忽略已有 checkpoint，从头训练。
RESUME_FROM_CHECKPOINT = False

# 0 表示不限制训练步数。
# 调试时可设为 1，只完成一次参数更新后就停止并保存 checkpoint。
MAX_TRAIN_STEPS = 0

RANDOM_SEED = 42


# ============================================================================
# 二、基础工具函数
# ============================================================================


def initialize_log_files(resume_from_checkpoint: bool) -> None:
    """在训练开始前准备日志文件。

    从头训练时清空旧日志，避免不同实验的数据混在一起。
    断点续训时保留旧日志，新的训练记录会继续追加到文件末尾。
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not resume_from_checkpoint:
        TRAIN_LOG_PATH.write_text("", encoding="utf-8")
        METRICS_LOG_PATH.write_text("", encoding="utf-8")


def log_message(message: str) -> None:
    """把同一条文本同时输出到控制台和 train.log。"""

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{current_time}] {message}"

    # 控制台可以让训练者实时观察运行状态。
    print(formatted_message, flush=True)

    # 文件日志用于训练结束后回看，使用追加模式避免覆盖之前的记录。
    TRAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(formatted_message + "\n")


def log_training_metrics(metrics: dict[str, int | float]) -> None:
    """把一次训练日志点的数值指标追加到 metrics.jsonl。"""

    metrics_record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        **metrics,
    }

    METRICS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_LOG_PATH.open("a", encoding="utf-8") as metrics_file:
        json.dump(metrics_record, metrics_file, ensure_ascii=False)
        metrics_file.write("\n")


def set_random_seed(seed: int) -> None:
    """固定 Python 和 PyTorch 随机种子，让重复实验尽可能一致。"""

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    """选择训练设备，并打印 GPU_ID 在程序内外的对应关系。"""

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(device)
        log_message(f"使用物理 GPU {GPU_ID}，程序内设备为 {device}：{gpu_name}")
        return device

    log_message("没有检测到可用 CUDA，训练将使用 CPU")
    return torch.device("cpu")


def resolve_autocast_dtype(device: torch.device) -> torch.dtype | None:
    """根据顶部配置和硬件能力决定混合精度的数据类型。

    返回 None 表示不开启 autocast，整个前向传播使用 float32。
    """

    if MIXED_PRECISION == "float32" or device.type != "cuda":
        return None

    if MIXED_PRECISION == "float16":
        return torch.float16

    if MIXED_PRECISION == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "当前显卡不支持 bfloat16，请把 MIXED_PRECISION 改成 float16 或 float32"
            )
        return torch.bfloat16

    raise ValueError(
        "MIXED_PRECISION 只支持 bfloat16、float16、float32，"
        f"当前值为 {MIXED_PRECISION!r}"
    )


def create_autocast_context(
    device: torch.device,
    autocast_dtype: torch.dtype | None,
):
    """创建前向传播所需的 autocast 上下文。"""

    if autocast_dtype is None:
        return nullcontext()

    return torch.autocast(
        device_type=device.type,
        dtype=autocast_dtype,
    )


def calculate_learning_rate(
    update_step: int,
    total_update_steps: int,
    warmup_steps: int,
    max_learning_rate: float,
    min_learning_rate_ratio: float,
) -> float:
    """计算当前参数更新步对应的学习率。

    训练开始先从较小学习率线性升到 max_learning_rate，降低随机初始化阶段
    梯度剧烈波动的风险；warmup 结束后，再使用余弦曲线缓慢下降。
    """

    if total_update_steps <= 0:
        raise ValueError("total_update_steps 必须大于 0")
    if update_step < 0:
        raise ValueError("update_step 不能小于 0")
    if not 0.0 <= min_learning_rate_ratio <= 1.0:
        raise ValueError("min_learning_rate_ratio 必须位于 [0, 1] 区间")

    if warmup_steps > 0 and update_step < warmup_steps:
        # update_step 从 0 开始，所以使用 update_step + 1，避免第一步学习率等于 0。
        warmup_progress = (update_step + 1) / warmup_steps
        return max_learning_rate * warmup_progress

    min_learning_rate = max_learning_rate * min_learning_rate_ratio
    # update_step 的有效范围是 0 到 total_update_steps - 1。
    # 减去 1 后，最后一个实际更新步的 decay_progress 正好等于 1。
    decay_steps = max(total_update_steps - warmup_steps - 1, 1)
    decay_progress = (update_step - warmup_steps) / decay_steps
    decay_progress = min(max(decay_progress, 0.0), 1.0)

    cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_learning_rate + cosine_ratio * (
        max_learning_rate - min_learning_rate
    )


def set_optimizer_learning_rate(optimizer: Optimizer, learning_rate: float) -> None:
    """把当前学习率写入优化器的所有参数组。"""

    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def count_model_parameters(model: nn.Module) -> tuple[int, int]:
    """返回模型总参数量和需要梯度的参数量。"""

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total_parameters, trainable_parameters


# ============================================================================
# 三、数据、模型和优化器初始化
# ============================================================================


def load_model_config() -> MiniMindConfig:
    """读取模型结构配置。"""

    if MODEL_CONFIG_PATH is None:
        return MiniMindConfig()
    return MiniMindConfig.from_json(MODEL_CONFIG_PATH)


def create_training_dataset(
    model_config: MiniMindConfig,
) -> Dataset:
    """加载 tokenizer，并构造预训练 Dataset。"""

    tokenizer = load_tokenizer(TOKENIZER_PATH)

    if len(tokenizer) != model_config.vocab_size:
        raise ValueError(
            "tokenizer 词表大小必须和模型配置一致，"
            f"当前 tokenizer={len(tokenizer)}, model={model_config.vocab_size}"
        )

    if tokenizer.pad_token_id != model_config.pad_token_id:
        raise ValueError(
            "tokenizer.pad_token_id 必须和模型配置一致，"
            f"当前 tokenizer={tokenizer.pad_token_id}, "
            f"model={model_config.pad_token_id}"
        )

    return PretrainDataset(
        data_path=DATA_PATH,
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LEN,
    )


def calculate_batches_per_epoch(dataset: Dataset) -> int:
    """根据 Dataset 大小和 batch size 计算一个 epoch 的 batch 数。"""

    if DROP_LAST:
        batches_per_epoch = len(dataset) // BATCH_SIZE
    else:
        batches_per_epoch = math.ceil(len(dataset) / BATCH_SIZE)

    if batches_per_epoch == 0:
        raise ValueError(
            "数据量不足以组成一个 batch，请减小 BATCH_SIZE 或关闭 DROP_LAST"
        )
    return batches_per_epoch


def create_epoch_dataloader(
    dataset: Dataset,
    epoch: int,
    start_batch_index: int,
) -> tuple[DataLoader, int]:
    """为指定 epoch 创建顺序确定、支持从中间恢复的 DataLoader。

    每个 epoch 都使用 RANDOM_SEED + epoch 生成固定随机顺序。
    checkpoint 记录下一个 batch 的编号，恢复时直接删除已经训练过的样本下标，
    不需要重新读取并 tokenize 前面的 batch。
    """

    batches_per_epoch = calculate_batches_per_epoch(dataset)
    if not 0 <= start_batch_index <= batches_per_epoch:
        raise ValueError(
            "start_batch_index 超出当前 epoch 范围，"
            f"当前值={start_batch_index}, batch 总数={batches_per_epoch}"
        )

    random_generator = torch.Generator()
    random_generator.manual_seed(RANDOM_SEED + epoch)
    shuffled_indices = torch.randperm(
        len(dataset),
        generator=random_generator,
    ).tolist()

    # checkpoint 只会在完整 batch、完整 optimizer step 后保存。
    # 因此可以用 batch 下标准确换算出需要跳过的样本数量。
    trained_sample_count = start_batch_index * BATCH_SIZE
    remaining_indices = shuffled_indices[trained_sample_count:]
    remaining_dataset = Subset(dataset, remaining_indices)

    dataloader = build_dataloader(
        dataset=remaining_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=DROP_LAST,
    )
    return dataloader, batches_per_epoch


def create_model_and_optimizer(
    model_config: MiniMindConfig,
    device: torch.device,
) -> tuple[MiniMindForCausalLM, Optimizer]:
    """创建模型和 AdamW 优化器。"""

    model = MiniMindForCausalLM(model_config)
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    return model, optimizer


# ============================================================================
# 四、checkpoint 保存和恢复
# ============================================================================


def save_checkpoint(
    checkpoint_path: Path,
    model: MiniMindForCausalLM,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    model_config: MiniMindConfig,
    epoch: int,
    next_batch_index: int,
    global_update_step: int,
) -> None:
    """原子保存完整训练状态，供中断后继续训练。

    先写入临时文件，完整写完后再替换正式文件。这样即使保存过程中断电，
    也不会把上一次可用的 checkpoint 覆盖成半个损坏文件。
    """

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "model_config": model_config.to_dict(),
        "epoch": epoch,
        "next_batch_index": next_batch_index,
        "global_update_step": global_update_step,
    }
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: MiniMindForCausalLM,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, int, int]:
    """恢复模型、优化器和混合精度状态。

    返回：开始 epoch、该 epoch 的开始 batch、已经完成的参数更新步数。
    """

    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = int(checkpoint["epoch"])
    start_batch_index = int(checkpoint["next_batch_index"])
    global_update_step = int(checkpoint["global_update_step"])
    return start_epoch, start_batch_index, global_update_step


def save_final_model(
    model: MiniMindForCausalLM,
    model_config: MiniMindConfig,
) -> None:
    """保存最终模型权重和对应配置。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_DIR / "model_weights.pt")
    model_config.save_json(OUTPUT_DIR / "config.json")


# ============================================================================
# 五、单个 epoch 的训练逻辑
# ============================================================================


def train_one_epoch(
    model: MiniMindForCausalLM,
    dataset: Dataset,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    model_config: MiniMindConfig,
    epoch: int,
    start_batch_index: int,
    global_update_step: int,
    total_update_steps: int,
    warmup_steps: int,
) -> tuple[int, bool]:
    """训练一个 epoch。

    返回更新后的 global_update_step，以及是否因为 MAX_TRAIN_STEPS 提前停止。
    """

    dataloader, batches_per_epoch = create_epoch_dataloader(
        dataset=dataset,
        epoch=epoch,
        start_batch_index=start_batch_index,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    epoch_start_time = time.time()
    loss_sum_since_log = 0.0
    batches_since_log = 0

    for local_batch_index, batch in enumerate(dataloader):
        # DataLoader 是从 start_batch_index 之后创建的，二者相加才是本 epoch
        # 中的真实 batch 编号。
        batch_index = start_batch_index + local_batch_index

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with create_autocast_context(device, autocast_dtype):
            output = model(
                input_ids=input_ids,
                labels=labels,
                use_cache=False,
            )

            if output.loss is None:
                raise RuntimeError("训练时传入了 labels，但模型没有返回 loss")
            raw_loss = output.loss

            if not torch.isfinite(raw_loss):
                raise FloatingPointError(
                    f"检测到非有限 loss：{raw_loss.item()}，训练已停止"
                )

            # 如果 epoch 最后剩余 batch 不足完整累积次数，就按实际剩余次数缩放。
            accumulation_window_start = (
                batch_index // GRADIENT_ACCUMULATION_STEPS
            ) * GRADIENT_ACCUMULATION_STEPS
            accumulation_window_size = min(
                GRADIENT_ACCUMULATION_STEPS,
                batches_per_epoch - accumulation_window_start,
            )
            loss_for_backward = raw_loss / accumulation_window_size

        # GradScaler 只在 float16 时真正放大梯度；bfloat16/float32 时相当于普通 backward。
        scaler.scale(loss_for_backward).backward()

        loss_sum_since_log += raw_loss.detach().item()
        batches_since_log += 1

        completed_batches = batch_index + 1
        accumulation_is_complete = (
            completed_batches % GRADIENT_ACCUMULATION_STEPS == 0
        )
        epoch_is_complete = completed_batches == batches_per_epoch

        # 只有完成一次梯度累积，或者到达 epoch 最后一个 batch，才真正更新模型参数。
        if not accumulation_is_complete and not epoch_is_complete:
            continue

        learning_rate = calculate_learning_rate(
            update_step=global_update_step,
            total_update_steps=total_update_steps,
            warmup_steps=warmup_steps,
            max_learning_rate=LEARNING_RATE,
            min_learning_rate_ratio=MIN_LEARNING_RATE_RATIO,
        )
        set_optimizer_learning_rate(optimizer, learning_rate)

        # 先把 float16 放大的梯度恢复，再执行梯度裁剪。
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=MAX_GRAD_NORM,
        )

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        global_update_step += 1

        should_log = (
            global_update_step % LOG_INTERVAL == 0
            or global_update_step == 1
            or epoch_is_complete
        )
        if should_log:
            average_loss = loss_sum_since_log / batches_since_log
            elapsed_seconds = time.time() - epoch_start_time
            log_message(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{completed_batches}/{batches_per_epoch}] "
                f"Update [{global_update_step}/{total_update_steps}] "
                f"loss={average_loss:.4f} "
                f"lr={learning_rate:.8f} "
                f"grad_norm={float(gradient_norm):.4f} "
                f"time={elapsed_seconds / 60:.1f}min"
            )

            # 结构化日志只保存数值，不需要后续再从文本字符串中解析。
            log_training_metrics(
                {
                    "epoch": epoch + 1,
                    "batch": completed_batches,
                    "batches_per_epoch": batches_per_epoch,
                    "update_step": global_update_step,
                    "total_update_steps": total_update_steps,
                    "loss": average_loss,
                    "learning_rate": learning_rate,
                    "grad_norm": float(gradient_norm),
                    "elapsed_seconds": elapsed_seconds,
                }
            )
            loss_sum_since_log = 0.0
            batches_since_log = 0

        reached_save_interval = global_update_step % SAVE_INTERVAL == 0
        reached_max_steps = (
            MAX_TRAIN_STEPS > 0
            and global_update_step >= MAX_TRAIN_STEPS
        )

        if reached_save_interval or reached_max_steps:
            save_checkpoint(
                checkpoint_path=OUTPUT_DIR / "latest_checkpoint.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                model_config=model_config,
                epoch=epoch,
                next_batch_index=completed_batches,
                global_update_step=global_update_step,
            )
            log_message(f"checkpoint 已保存：{OUTPUT_DIR / 'latest_checkpoint.pt'}")

        if reached_max_steps:
            return global_update_step, True

    return global_update_step, False


# ============================================================================
# 六、完整预训练入口
# ============================================================================


def main() -> None:
    """依次完成初始化、恢复状态、多轮训练和最终权重保存。"""

    initialize_log_files(RESUME_FROM_CHECKPOINT)
    log_message("=" * 72)
    if RESUME_FROM_CHECKPOINT:
        log_message("开始断点续训，日志将追加到已有文件")
    else:
        log_message("开始新的预训练任务，旧日志已清空")

    set_random_seed(RANDOM_SEED)
    device = select_device()
    autocast_dtype = resolve_autocast_dtype(device)
    model_config = load_model_config()
    dataset = create_training_dataset(model_config)
    model, optimizer = create_model_and_optimizer(model_config, device)

    # float16 的有效数值范围较小，需要 GradScaler；bfloat16 和 float32 不需要。
    scaler_enabled = device.type == "cuda" and autocast_dtype == torch.float16
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=scaler_enabled,
    )

    batches_per_epoch = calculate_batches_per_epoch(dataset)
    update_steps_per_epoch = math.ceil(
        batches_per_epoch / GRADIENT_ACCUMULATION_STEPS
    )
    total_update_steps = EPOCHS * update_steps_per_epoch
    if MAX_TRAIN_STEPS > 0:
        total_update_steps = min(total_update_steps, MAX_TRAIN_STEPS)
    warmup_steps = int(total_update_steps * WARMUP_RATIO)

    total_parameters, trainable_parameters = count_model_parameters(model)
    log_message(f"训练样本数：{len(dataset):,}")
    log_message(f"每个 epoch 的 batch 数：{batches_per_epoch:,}")
    log_message(f"计划参数更新次数：{total_update_steps:,}")
    log_message(f"模型总参数量：{total_parameters / 1_000_000:.2f}M")
    log_message(f"可训练参数量：{trainable_parameters / 1_000_000:.2f}M")
    log_message(f"混合精度：{autocast_dtype or torch.float32}")

    start_epoch = 0
    start_batch_index = 0
    global_update_step = 0
    checkpoint_path = OUTPUT_DIR / "latest_checkpoint.pt"

    if RESUME_FROM_CHECKPOINT:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"找不到续训 checkpoint：{checkpoint_path}")
        start_epoch, start_batch_index, global_update_step = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        log_message(
            "已恢复训练状态："
            f"epoch={start_epoch + 1}, "
            f"next_batch={start_batch_index}, "
            f"update_step={global_update_step}"
        )

    if global_update_step >= total_update_steps:
        log_message("checkpoint 中的训练步数已经达到本次计划，无需继续训练")
        return

    for epoch in range(start_epoch, EPOCHS):
        current_start_batch = start_batch_index if epoch == start_epoch else 0

        global_update_step, stopped_early = train_one_epoch(
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            autocast_dtype=autocast_dtype,
            model_config=model_config,
            epoch=epoch,
            start_batch_index=current_start_batch,
            global_update_step=global_update_step,
            total_update_steps=total_update_steps,
            warmup_steps=warmup_steps,
        )
        if stopped_early:
            save_final_model(model, model_config)
            log_message(f"已达到 MAX_TRAIN_STEPS，模型保存到：{OUTPUT_DIR}")
            return

        # 一个 epoch 完整结束后，下次恢复应该从下一个 epoch 的第 0 个 batch 开始。
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            model_config=model_config,
            epoch=epoch + 1,
            next_batch_index=0,
            global_update_step=global_update_step,
        )
        log_message(f"Epoch {epoch + 1} 完成，checkpoint 已保存")

    save_final_model(model, model_config)
    log_message(f"预训练完成，最终模型保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()

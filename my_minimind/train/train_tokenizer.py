import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


# ==================== 默认训练参数 ====================

# tokenizer 预训练语料。
DEFAULT_DATA_PATH = "datasets/pretrain_t2t_mini.jsonl"

# 是否额外加入 SFT mini 数据训练 tokenizer。
# 开启后，会从 DEFAULT_SFT_DATA_PATH 中随机采样 DEFAULT_SFT_MAX_LINES 条对话文本。
DEFAULT_USE_SFT_DATA = True

# tokenizer SFT 语料。
DEFAULT_SFT_DATA_PATH = "datasets/sft_t2t_mini.jsonl"



# tokenizer 输出目录。
DEFAULT_SAVE_DIR = "my_minimind/tokenizer"

# 使用哪张显卡，可选 "0" 或 "1"。
# tokenizer 训练主要使用 CPU，这里只是统一设置运行环境。
DEFAULT_GPU_ID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = DEFAULT_GPU_ID

#tokenizer词表大小是 6400
DEFAULT_VOCAB_SIZE = 6400

# BPE 合并时要求 pair 至少出现多少次。
# 值越大，越多低频 pair 会被过滤，内存压力越小，但可能损失少见词/符号覆盖。
DEFAULT_MIN_FREQUENCY = 2

# BPE 允许生成的单个 token 的最大长度。
# 限制过长 token 可以减少 URL、乱码、长数字、重复字符带来的 pair 统计压力。
DEFAULT_MAX_TOKEN_LENGTH = 32

# MiniMind tokenizer 预留了 36 个特殊 token 位置。
# 这些 token 既服务聊天格式，也为多模态、工具调用、思考标签预留接口。
DEFAULT_SPECIAL_TOKENS_NUM = 36

# 随机采样多少条有效文本训练 tokenizer。
# 不建议默认读取完整 1G+ 数据文件：BPE 在 Count pairs 阶段会占用大量内存，
# 全量训练容易被系统 OOM killer 直接杀掉。
# 训练 tokenizer 只需要有代表性的文本子集，先随机采样 50000 条更稳。
# 如果你确认机器内存足够，可以改成 0 表示读取整个文件。
DEFAULT_MAX_LINES = 640000

# 随机采样多少条 SFT 有效文本。0 表示读取整个 SFT 文件。
DEFAULT_SFT_MAX_LINES = 100000


# 随机采样种子。
# 固定种子可以保证每次训练 tokenizer 时抽到的样本一致，方便复现实验。
DEFAULT_RANDOM_SEED = 42

# 普通预训练 JSONL 中的文本字段名。
DEFAULT_TEXT_FIELD = "text"

# SFT JSONL 中的对话字段名。
# 当前默认不用 SFT 数据，但保留这个字段是为了脚本兼容 MiniMind 的对话数据格式。
DEFAULT_CONVERSATION_FIELD = "conversations"

# 从MiniMind tokenizer 配置中复用 chat_template。
# 这样自己训练 tokenizer 后，后续 SFT 对话拼接格式仍然和原项目一致。
DEFAULT_CHAT_TEMPLATE_FROM = "original_project/model/tokenizer_config.json"

# 是否在训练完成后做一次 encode/decode 冒烟评测。
DEFAULT_EVAL = True

# 这一组 token 会在 tokenizer_config.json 中标记为 special=True。
# 它们通常不会被普通文本拆分，适合表示结构边界或特殊模态占位符。
BASE_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<|audio_pad|>",
    "<tts_pad>",
    "<tts_text_bos>",
    "<tts_text_eod>",
    "<tts_text_bos_single>",
]

# 这些 token 会参与训练器的 special token 预留，但后面会在 tokenizer.json
# 里标记成 special=False。这样它们可以稳定保留为单 token，同时又能作为普通内容解码。
ADDITIONAL_TOKENS = [
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 my_minimind 自己的 tokenizer")

    # 支持 JSONL 数据。每行可以是 {"text": "..."}，也可以是 MiniMind SFT 风格的
    # {"conversations": [{"role": "...", "content": "..."}]}。
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="用于训练 tokenizer 的 JSONL 文件")

    # 是否额外使用 SFT mini 数据。
    parser.add_argument("--use_sft_data", action="store_true", help="启用 SFT mini 数据参与 tokenizer 训练")
    parser.add_argument("--no_sft_data", action="store_true", help="禁用 SFT mini 数据参与 tokenizer 训练")

    # SFT mini 数据路径。
    parser.add_argument("--sft_data_path", type=str, default=DEFAULT_SFT_DATA_PATH, help="SFT mini JSONL 文件")

    # SFT mini 随机采样条数。
    parser.add_argument("--sft_max_lines", type=int, default=DEFAULT_SFT_MAX_LINES, help="随机采样多少条 SFT 有效文本，0 表示全部读取")

    # tokenizer 输出目录，会写入 tokenizer.json、vocab.json、merges.txt、tokenizer_config.json。
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR, help="tokenizer 保存目录")

    # 对齐 MiniMind-3，默认 6400。
    parser.add_argument("--vocab_size", type=int, default=DEFAULT_VOCAB_SIZE, help="词表大小")

    # BPE 低频过滤阈值。
    parser.add_argument("--min_frequency", type=int, default=DEFAULT_MIN_FREQUENCY, help="BPE pair 最小出现频次")

    # BPE 单个 token 最大长度。
    parser.add_argument("--max_token_length", type=int, default=DEFAULT_MAX_TOKEN_LENGTH, help="BPE 单个 token 最大长度")

    # 训练时随机采样多少条有效文本。0 表示读取全部。
    parser.add_argument("--max_lines", type=int, default=DEFAULT_MAX_LINES, help="随机采样多少条有效文本，0 表示全部读取")

    # 随机采样种子。
    parser.add_argument("--random_seed", type=int, default=DEFAULT_RANDOM_SEED, help="随机采样种子")


    # 普通预训练数据的文本字段名。
    parser.add_argument("--text_field", type=str, default=DEFAULT_TEXT_FIELD, help="普通文本 JSONL 的字段名")

    # SFT 数据的对话字段名。
    parser.add_argument("--conversation_field", type=str, default=DEFAULT_CONVERSATION_FIELD, help="对话 JSONL 的字段名")

    # 特殊 token 总数。保持 36 可以和 MiniMind 的预留习惯一致。
    parser.add_argument("--special_tokens_num", type=int, default=DEFAULT_SPECIAL_TOKENS_NUM, help="特殊 token 预留总数")

    # 从原始 MiniMind tokenizer_config.json 里复用 chat_template。
    # 这样自己训练 tokenizer 后，SFT 阶段仍然能使用同一种对话拼接格式。
    parser.add_argument(
        "--chat_template_from",
        type=str,
        default=DEFAULT_CHAT_TEMPLATE_FROM,
        help="从哪个 tokenizer_config.json 复制 chat_template",
    )

    # 训练完成后做一次简单编码、解码和压缩率检查。
    parser.add_argument("--eval", action="store_true", help="训练后做 tokenizer 冒烟评测")
    parser.add_argument("--no_eval", action="store_true", help="关闭训练后的 tokenizer 冒烟评测")
    return parser.parse_args()


def build_special_tokens(special_tokens_num: int) -> tuple[list[str], list[str], list[str]]:
    """构造 tokenizer 需要预留的特殊 token 列表。"""

    used_tokens = BASE_SPECIAL_TOKENS + ADDITIONAL_TOKENS
    if special_tokens_num < len(used_tokens):
        raise ValueError(
            f"special_tokens_num={special_tokens_num} 太小，至少需要 {len(used_tokens)}"
        )

    # buffer token 是预留坑位。
    # 以后如果要加入新结构 token，可以尽量复用这些位置，减少破坏已有格式的风险。
    buffer_count = special_tokens_num - len(used_tokens)
    buffer_tokens = [f"<|buffer{i}|>" for i in range(1, buffer_count + 1)]
    all_special_tokens = used_tokens + buffer_tokens
    return BASE_SPECIAL_TOKENS, ADDITIONAL_TOKENS, all_special_tokens


def iter_training_texts(
    data_path: Path,
    text_field: str,
    conversation_field: str,
    max_lines: int,
    random_seed: int,
) -> Iterable[str]:
    """逐行读取 JSONL，并产出 tokenizer 训练文本。

    max_lines > 0 时，不再取文件前 N 行，而是对整个文件做随机采样。
    这里使用 reservoir sampling：只在内存里保留 max_lines 条文本，同时保证
    每条有效文本被抽中的概率基本一致。
    """

    if not data_path.exists():
        raise FileNotFoundError(f"找不到 tokenizer 训练数据：{data_path}")

    rng = random.Random(random_seed)
    reservoir: list[str] = []
    valid_text_count = 0

    with data_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                # 训练 tokenizer 时允许跳过坏行，避免一行脏数据中断整个训练。
                continue

            text = extract_text(sample, text_field, conversation_field)
            if not text:
                continue

            # max_lines <= 0 表示不采样，直接把所有有效文本流式产出。
            if max_lines <= 0:
                yield text
                continue

            valid_text_count += 1
            if len(reservoir) < max_lines:
                reservoir.append(text)
                continue

            # reservoir sampling 核心：
            # 第 k 条有效文本有 max_lines/k 的概率进入样本池。
            replace_index = rng.randint(1, valid_text_count)
            if replace_index <= max_lines:
                reservoir[replace_index - 1] = text

    if max_lines > 0:
        print(
            f"随机采样完成：有效文本 {valid_text_count} 条，"
            f"实际采样 {len(reservoir)} 条，random_seed={random_seed}"
        )
        for text in reservoir:
            yield text


def extract_text(sample: dict[str, Any], text_field: str, conversation_field: str) -> str:
    """从一条 JSON 样本中抽取可用于 tokenizer 训练的纯文本。"""

    # 普通预训练格式：{"text": "..."}。
    if text_field in sample and sample[text_field]:
        return str(sample[text_field])

    # MiniMind SFT 格式：{"conversations": [{"role": "...", "content": "..."}]}。
    conversations = sample.get(conversation_field)
    if isinstance(conversations, list):
        contents = []
        for item in conversations:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if content:
                contents.append(str(content))
        return "\n".join(contents)

    return ""


def chain_text_iterators(iterators: list[Iterable[str]]) -> Iterable[str]:
    """按顺序合并多个文本迭代器。

    这里不用先把所有文本读进列表，而是一个迭代器读完再读下一个。
    这样可以把预训练采样和 SFT 采样组合起来，同时控制内存占用。
    """

    for iterator in iterators:
        yield from iterator


def load_chat_template(config_path: Path) -> str:
    """从已有 tokenizer_config.json 中读取 chat_template。"""

    if not config_path.exists():
        raise FileNotFoundError(f"找不到 chat_template 来源文件：{config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    chat_template = config.get("chat_template")
    if not chat_template:
        raise ValueError(f"{config_path} 中没有 chat_template 字段")
    return str(chat_template)


def patch_added_token_flags(tokenizer_json_path: Path, base_special_tokens: list[str]) -> None:
    """修正 tokenizer.json 中 added_tokens 的 special 标记。"""

    with tokenizer_json_path.open("r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    for token_info in tokenizer_data.get("added_tokens", []):
        # MiniMind 的做法是：基础结构 token 标记为 special=True；
        # tool/thinking/buffer 这类额外 token 保持单 token，但 special=False。
        token_info["special"] = token_info.get("content") in base_special_tokens

    with tokenizer_json_path.open("w", encoding="utf-8") as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def build_tokenizer_config(
    tokenizer: Tokenizer,
    all_special_tokens: list[str],
    base_special_tokens: list[str],
    chat_template: str,
) -> dict[str, Any]:
    """构造 Hugging Face AutoTokenizer 能识别的 tokenizer_config.json。"""

    added_tokens_decoder = {}
    for token in all_special_tokens:
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"特殊 token 没有进入词表：{token}")

        added_tokens_decoder[str(token_id)] = {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": token in base_special_tokens,
        }

    return {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": added_tokens_decoder,
        "additional_special_tokens": [t for t in base_special_tokens if t != "<|endoftext|>"],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 131072,
        "pad_token": "<|endoftext|>",
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|endoftext|>",
        "image_token": "<|image_pad|>",
        "audio_token": "<|audio_pad|>",
        "video_token": "<|video_pad|>",
        "vision_bos_token": "<|vision_start|>",
        "vision_eos_token": "<|vision_end|>",
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
        "chat_template": chat_template,
    }


def train_tokenizer(args: argparse.Namespace) -> Path:
    """训练并保存 ByteLevel BPE tokenizer。"""

    data_path = Path(args.data_path)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    use_sft_data = DEFAULT_USE_SFT_DATA
    if args.use_sft_data:
        use_sft_data = True
    if args.no_sft_data:
        use_sft_data = False

    base_special_tokens, _, all_special_tokens = build_special_tokens(args.special_tokens_num)
    chat_template = load_chat_template(Path(args.chat_template_from))

    # BPE 是大模型常用的子词算法之一。
    # 它会从小片段开始，逐步合并高频相邻片段，直到达到目标词表大小。
    tokenizer = Tokenizer(models.BPE())

    # ByteLevel 预分词会把文本映射到字节层面。
    # 优点是几乎任何 Unicode 文本都能被编码，中文、英文、代码、符号混合时更稳。
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=all_special_tokens,
        max_token_length=args.max_token_length,
    )

    text_iterators = [
        iter_training_texts(
            data_path=data_path,
            text_field=args.text_field,
            conversation_field=args.conversation_field,
            max_lines=args.max_lines,
            random_seed=args.random_seed,
        )
    ]

    if use_sft_data:
        text_iterators.append(
            iter_training_texts(
                data_path=Path(args.sft_data_path),
                text_field=args.text_field,
                conversation_field=args.conversation_field,
                max_lines=args.sft_max_lines,
                random_seed=args.random_seed + 1,
            )
        )

    print(f"预训练数据：{data_path}，随机采样条数：{args.max_lines}")
    if use_sft_data:
        print(f"SFT 数据：{args.sft_data_path}，随机采样条数：{args.sft_max_lines}")
    else:
        print("SFT 数据：未启用")

    texts = chain_text_iterators(text_iterators)
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # ByteLevel decoder 负责把 byte-level token 还原回正常字符串。
    tokenizer.decoder = decoders.ByteLevel()

    # 再显式添加基础特殊 token，保持和 MiniMind 原脚本一致。
    tokenizer.add_special_tokens(base_special_tokens)

    tokenizer_json_path = save_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json_path))

    # 同时保存 vocab.json 和 merges.txt，便于人工检查 BPE 合并规则。
    tokenizer.model.save(str(save_dir))

    patch_added_token_flags(tokenizer_json_path, base_special_tokens)

    tokenizer_config = build_tokenizer_config(
        tokenizer=tokenizer,
        all_special_tokens=all_special_tokens,
        base_special_tokens=base_special_tokens,
        chat_template=chat_template,
    )
    with (save_dir / "tokenizer_config.json").open("w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"tokenizer 训练完成，保存目录：{save_dir}")
    print(f"目标词表大小：{args.vocab_size}")
    print(f"实际词表大小：{tokenizer.get_vocab_size()}")
    return save_dir


def eval_tokenizer(tokenizer_dir: Path) -> None:
    """训练后做一个简单冒烟评测。"""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

    test_texts = [
        "你好，我正在手搓一个小语言模型。",
        "Large language models use tokenizers to convert text into token ids.",
        "Python、PyTorch、CUDA 和 tokenizer 都是训练链路中的关键组件。",
    ]

    print("-" * 80)
    print(f"tokenizer 词表长度：{len(tokenizer)}")
    for text in test_texts:
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
        ratio = len(text) / max(len(token_ids), 1)
        print(f"文本：{text}")
        print(f"token 数：{len(token_ids)}，字符/token 压缩率：{ratio:.2f}")
        print(f"解码一致：{decoded == text}")
        print("-" * 80)


def main() -> None:
    args = parse_args()
    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    tokenizer_dir = train_tokenizer(args)
    should_eval = DEFAULT_EVAL
    if args.eval:
        should_eval = True
    if args.no_eval:
        should_eval = False

    if should_eval:
        eval_tokenizer(tokenizer_dir)


if __name__ == "__main__":
    main()

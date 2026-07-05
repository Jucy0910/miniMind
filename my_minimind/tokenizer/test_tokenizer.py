from pathlib import Path

from transformers import AutoTokenizer


# 当前脚本放在 tokenizer 目录下，所以父目录就是 tokenizer 文件目录。
TOKENIZER_DIR = Path(__file__).resolve().parent


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def test_basic_info(tokenizer) -> None:
    """检查 tokenizer 的基础配置。"""

    print_section("基础信息")
    print(f"tokenizer 目录: {TOKENIZER_DIR}")
    print(f"len(tokenizer): {len(tokenizer)}")
    print(f"tokenizer.vocab_size: {tokenizer.vocab_size}")
    print(f"bos_token: {tokenizer.bos_token!r}, bos_token_id: {tokenizer.bos_token_id}")
    print(f"eos_token: {tokenizer.eos_token!r}, eos_token_id: {tokenizer.eos_token_id}")
    print(f"pad_token: {tokenizer.pad_token!r}, pad_token_id: {tokenizer.pad_token_id}")
    print(f"unk_token: {tokenizer.unk_token!r}, unk_token_id: {tokenizer.unk_token_id}")


def test_special_tokens(tokenizer) -> None:
    """检查 MiniMind 风格特殊 token 是否能稳定编码成单个 token。"""

    print_section("特殊 token")
    special_tokens = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<|image_pad|>",
        "<|audio_pad|>",
        "<|video_pad|>",
    ]

    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer.encode(token, add_special_tokens=False)
        print(f"{token!r:18} id={token_id!s:>5} encode={encoded}")


def test_encode_decode(tokenizer) -> None:
    """检查普通文本能否 encode 后再 decode 回来。"""

    print_section("encode/decode")
    test_texts = [
        "你好，我正在手搓一个小语言模型。",
        "Large language models use tokenizers to convert text into token ids.",
        "Python、PyTorch、CUDA 和 tokenizer 都是训练链路中的关键组件。",
        "代码示例：for i in range(3): print(i)",
    ]

    for text in test_texts:
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
        print(f"原文: {text}")
        print(f"token 数: {len(input_ids)}")
        print(f"input_ids 前 30 个: {input_ids[:30]}")
        print(f"解码一致: {decoded == text}")
        print(f"解码: {decoded}")
        print("-" * 80)


def test_chat_template(tokenizer) -> None:
    """检查 chat_template 能否把 messages 渲染成模型输入文本。"""

    print_section("chat_template")
    messages = [
        {"role": "system", "content": "你是一个简洁、可靠的 AI 助手。"},
        {"role": "user", "content": "请用一句话解释 BPE tokenizer。"},
        {"role": "assistant", "content": "BPE tokenizer 会通过反复合并高频片段，把文本切成子词 token。"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    print("渲染后的 prompt:")
    print(prompt)

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    print(f"chat_template tokenize 后 token 数: {len(input_ids)}")
    print(f"input_ids 前 50 个: {input_ids[:50]}")


def test_generation_prompt(tokenizer) -> None:
    """检查推理时 add_generation_prompt 的结尾格式。"""

    print_section("add_generation_prompt")
    messages = [
        {"role": "user", "content": "你好，请介绍一下你自己。"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print(prompt)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    test_basic_info(tokenizer)
    test_special_tokens(tokenizer)
    test_encode_decode(tokenizer)
    test_chat_template(tokenizer)
    test_generation_prompt(tokenizer)


if __name__ == "__main__":
    main()


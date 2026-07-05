from pathlib import Path
from typing import Any


def load_tokenizer(tokenizer_path: str | Path, **kwargs: Any):
    """从本地路径加载 Hugging Face 兼容的 tokenizer。
    原始 MiniMind 项目在 ``original_project/model`` 下提供了 tokenizer。
    这里把 tokenizer 加载逻辑单独封装起来，后续如果换成自己训练的 tokenizer，
    就不用改数据集和训练代码。
    """
    # transformers 是 tokenizer 的实际提供方。
    # 这里放在函数内部导入，是为了让只看配置、模型结构时不强制依赖 transformers。
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "加载 MiniMind tokenizer 需要安装 transformers。"
            "请先安装项目依赖。"
        ) from exc

    # AutoTokenizer 会根据 tokenizer_path 里的 tokenizer.json、
    # tokenizer_config.json 等文件自动构造 tokenizer。
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), **kwargs)

    # 有些 tokenizer 没有显式 pad token。
    # 训练时 batch 需要 padding，所以这里兜底复用 eos token 作为 pad token。
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

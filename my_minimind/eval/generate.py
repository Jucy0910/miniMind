import argparse


def parse_args() -> argparse.Namespace:
    # 生成脚本的命令行入口。
    # 后续会支持从 checkpoint 加载模型，并对 prompt 进行续写。
    parser = argparse.ArgumentParser(description="第一阶段生成入口")

    # 模型权重路径，后续可能是 checkpoint 目录或单个 .pt/.pth 文件。
    parser.add_argument("--load_from", type=str, required=False, default="")

    # 用户输入的提示词。
    parser.add_argument("--prompt", type=str, default="你好")
    return parser.parse_args()


def main() -> None:
    # 第一阶段只解析参数，不做真实生成。
    # 真正生成需要 tokenizer、模型结构、checkpoint 加载和采样策略都准备好。
    args = parse_args()
    raise NotImplementedError(
        "生成逻辑会在后续阶段实现。"
        f"收到 load_from={args.load_from!r}, prompt={args.prompt!r}。"
    )


if __name__ == "__main__":
    main()

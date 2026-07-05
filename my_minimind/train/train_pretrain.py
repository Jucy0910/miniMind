import argparse
import json

from my_minimind.configs import MiniMindConfig, TrainConfig


def parse_args() -> argparse.Namespace:
    # argparse 负责把命令行参数解析成 Python 对象。
    # 例如：
    #   python -m my_minimind.train.train_pretrain --dry_run
    # 会让 args.dry_run 等于 True。
    parser = argparse.ArgumentParser(description="第一阶段 MiniMind 预训练入口")

    # 预训练数据路径。第一阶段先不真正训练，所以默认留空。
    parser.add_argument("--data_path", type=str, default="")

    # tokenizer 默认复用原始项目中的 tokenizer。
    # 后续我们会逐步理解 tokenizer 文件，并可以手动训练自己的 tokenizer。
    parser.add_argument("--tokenizer_path", type=str, default="original_project/model")

    # checkpoint 或中间结果保存目录。
    parser.add_argument("--save_dir", type=str, default="my_minimind_out")

    # 下面几个参数直接影响模型结构。
    # 命令行传参可以覆盖默认配置，方便做小模型实验。
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    # 模型结构支持的最大位置长度，对齐 MiniMind-3 的 max_position_embeddings=32768。
    parser.add_argument("--max_position_embeddings", type=int, default=32768)

    # 训练时实际截断到的样本长度。
    # 它可以小于 max_position_embeddings，方便早期用较少显存做实验。
    parser.add_argument("--max_seq_len", type=int, default=512)

    # dry_run 用于只打印配置，不执行训练。
    # 第一阶段用它验证入口和配置对象是否正常。
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    # 解析命令行参数。
    args = parse_args()

    # 组装模型配置。
    # 这里先只暴露几个最常调的参数，其余使用 MiniMindConfig 默认值。
    model_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_position_embeddings,
    )

    # 训练配置目前只绑定 max_seq_len。
    # 后续实现训练循环时会逐步使用 batch_size、lr、epochs 等字段。
    train_config = TrainConfig(max_seq_len=args.max_seq_len)

    if args.dry_run:
        # 用 JSON 格式打印，便于观察每个字段和后续保存到文件。
        print("model_config:")
        print(json.dumps(model_config.to_dict(), ensure_ascii=False, indent=2))
        print("train_config:")
        print(json.dumps(train_config.__dict__, ensure_ascii=False, indent=2))
        return

    # 第一阶段故意不写假训练逻辑。
    # 只有 Transformer 真正实现后，训练循环才有意义。
    raise NotImplementedError(
        "第一阶段只定义项目骨架。"
        "真正的预训练循环会在 Transformer 模型实现后补上。"
    )


if __name__ == "__main__":
    main()

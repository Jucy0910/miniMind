#!/usr/bin/env bash
set -euo pipefail

# 这个脚本用于下载 MiniMind 数据集中的四个核心文本数据：
# 1. 预训练 mini 版
# 2. 预训练完整版
# 3. SFT mini 版
# 4. SFT 完整版
#
# 数据会保存到仓库根目录的 datasets/。
# datasets/ 已经被 .gitignore 忽略，避免误提交 GB 级数据文件。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/datasets"
LOG_FILE="${DATA_DIR}/download_minimind_datasets.log"

mkdir -p "${DATA_DIR}"

# 所有输出同时写入终端和日志文件，方便后台下载时查看进度和错误。
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "开始下载 MiniMind 数据集"
echo "保存目录：${DATA_DIR}"
echo "日志文件：${LOG_FILE}"
echo "开始时间：$(date '+%Y-%m-%d %H:%M:%S')"
echo

download_one() {
  local file_name="$1"
  echo "============================================================"
  echo "开始下载：${file_name}"
  echo "时间：$(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"

  conda run -n miniconda modelscope download \
    --dataset gongjy/minimind_dataset \
    "${file_name}" \
    --local_dir "${DATA_DIR}"

  echo
  echo "完成下载：${file_name}"
  echo "当前文件大小："
  du -h "${DATA_DIR}/${file_name}" || true
  echo
}

download_one "pretrain_t2t_mini.jsonl"
download_one "pretrain_t2t.jsonl"
download_one "sft_t2t_mini.jsonl"
download_one "sft_t2t.jsonl"

echo "============================================================"
echo "全部下载完成"
echo "结束时间：$(date '+%Y-%m-%d %H:%M:%S')"
echo "datasets 目录大小："
du -sh "${DATA_DIR}"
echo "============================================================"


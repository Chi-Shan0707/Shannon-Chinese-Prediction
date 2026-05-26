"""
样本选取脚本：从 yuhuanstudio/wikipedia-pretrain-zh 选 100 段中文文本

用法:
  conda activate qwenenv
  python pj/data/sample_selection.py --num_samples 120 --min_len 50 --seed 42
"""

import argparse
import random
import re
from pathlib import Path

from datasets import load_dataset


def is_good_sample(text: str) -> bool:
    """只保留纯中文文本：不含拉丁字母、数字、特殊符号"""
    if len(text) < 50:
        return False
    # 如果 90%+ 字符是 CJK 或中文标点，则接受
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3000' <= c <= '\u303f' or '\uff00' <= c <= '\uffef')
    if cjk_count / len(text) < 0.9:
        return False
    # 排除过短或过长
    if len(text) > 1000:
        text = text[:1000]
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=120)
    parser.add_argument("--min_len", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(__file__).parent
    output_path = args.output or (out_dir / "chinese_samples.txt")

    random.seed(args.seed)

    print("Loading dataset (streaming)...")
    ds = load_dataset("yuhuanstudio/wikipedia-pretrain-zh", split="train", streaming=True)

    sampled = []
    for i, row in enumerate(ds):
        text = row["text"].strip()
        if is_good_sample(text):
            sampled.append(text)
            if len(sampled) >= args.num_samples:
                break
        if i % 10000 == 0:
            print(f"  Scanned {i} rows, collected {len(sampled)}/{args.num_samples}")

    # Trim to required length
    sampled = [s[:args.min_len * 5] for s in sampled]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Chinese test samples from yuhuanstudio/wikipedia-pretrain-zh\n")
        f.write(f"# Total: {len(sampled)} samples\n")
        f.write(f"# Min length: {min(len(s) for s in sampled)} chars\n")
        f.write(f"# Max length: {max(len(s) for s in sampled)} chars\n\n")
        for i, s in enumerate(sampled, 1):
            f.write(f"=== Sample {i} ===\n{s}\n\n")

    print(f"\nDone: {len(sampled)} samples saved to {output_path}")
    print(f"  Length range: {min(len(s) for s in sampled)}-{max(len(s) for s in sampled)} chars")


if __name__ == "__main__":
    main()

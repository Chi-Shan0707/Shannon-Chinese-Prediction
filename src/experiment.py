"""
Shannon 汉语预测实验主脚本

流程:
  1. 从 data/chinese_samples.txt 加载测试样本
  2. 对每个样本的每个位置，取前 N 字作为条件 (N=0,1,2,3,5,10,20,50)
  3. 用 Qwen3-0.6B 获取下一个汉字的概率分布
  4. 计算 Shannon 熵上下界
  5. 汇总输出

用法:
  conda activate qwenenv
  python pj/src/experiment.py --output pj/reports/results.json

注意: 需要 GPU，首次运行约 5-10 分钟 (120 样本 × 8 个 N 值)
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from local_model import CharModel
from entropy_calculator import all_bounds

N_VALUES = [0, 1, 2, 3, 5, 10, 20, 50]
SAMPLES_PATH = "pj/data/chinese_samples.txt"


def load_samples(path: str) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    samples = re.findall(r"=== Sample \d+ ===\n(.+?)(?=\n===|\Z)", text, re.DOTALL)
    return [s.strip() for s in samples if len(s.strip()) >= 50]


def run_experiment(samples: list[str], model: CharModel) -> dict:
    results = {n: {"lower": [], "middle": []} for n in N_VALUES}

    for sid, sample in enumerate(samples, 1):
        for n in N_VALUES:
            if n == 0:
                # 零阶：模型的无条件先验（对所有位置相同）
                chars, probs = model.predict("")
                low, mid = all_bounds(probs)
                for _ in range(len(sample)):
                    results[n]["lower"].append(low)
                    results[n]["middle"].append(mid)
            else:
                # 逐位置滑动预测
                for pos in range(n, len(sample)):
                    prefix = sample[pos - n:pos]
                    chars, probs = model.predict(prefix)
                    low, mid = all_bounds(probs)
                    results[n]["lower"].append(low)
                    results[n]["middle"].append(mid)

        if sid % 20 == 0:
            print(f"  Processed {sid}/{len(samples)} samples")

    # 求平均
    summary = {}
    for n in N_VALUES:
        summary[n] = {
            k: (sum(v) / len(v)) if v else 0.0
            for k, v in results[n].items()
        }
    return {"by_N": summary, "N_values": N_VALUES}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="pj/reports/results.json")
    args = parser.parse_args()

    print("Loading test samples...")
    samples = load_samples(SAMPLES_PATH)
    print(f"  {len(samples)} samples loaded")

    print("Loading model (this may take a moment)...")
    model = CharModel()
    print(f"  Model on {model.device}, {len(model.char_ids)} valid char tokens")

    print("Running experiment...")
    t0 = time.time()
    results = run_experiment(samples, model)
    elapsed = time.time() - t0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {out_path} ({elapsed:.1f}s)")

    print("\n=== Summary (bits/char) ===")
    print(f"{'N':>5}  {'Lower':>8}  {'Middle':>8}")
    for n in N_VALUES:
        r = results["by_N"][n]
        print(f"{n:>5}  {r['lower']:>8.4f}  {r['middle']:>8.4f}")


if __name__ == "__main__":
    main()

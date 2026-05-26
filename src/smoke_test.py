"""
Quick smoke test: load model, pick a few text snippets, compute entropy bounds.

Usage:
  conda activate qwenenv
  python pj/src/smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from local_model import CharModel
from entropy_calculator import all_bounds

CLEAN_DIR = Path(__file__).parent.parent / "data" / "clean"

SNIPPETS = [
    ("人类简史", '大约在135亿年前，经过所谓的\u201c大爆炸\u201d之后，宇宙的物质、能量、时间和空间才成了现在的样子。宇宙的这些基本特征，就成了\u201c物理学\u201d。'),
    ("俗世奇人", "码头上的人，全是硬碰硬。手艺人靠的是手，手上就必得有绝活。有绝活的，吃荤，亮堂，站在大街中央"),
    ("KFC", "人生低谷时，总告诉自己会好的。直到今天才懂：会好的，是因为疯狂星期四，v我50，吃顿好的"),
    ("KFC2", "我本是五百强公司的老总，却被诡计多端的奸人所害！亲信弃我！家人逐我！甚至清空我的股份，变卖我的家产！重来一生，我只想夺回我失去的一切！今天疯狂星期四"),
    ("新闻-矿难", "2026年5月22日19时29分，山西通洲集团留神峪煤矿发生瓦斯爆炸事故，截至发稿时已造成82人死亡，救援工作仍在持续中"),
    ("新闻-脑机", "这位28岁的受试者因高位截瘫已卧床8年，肩部以下完全不能活动。但在植入脑机接口系统后，他不仅能用意念控制轮椅、机械手，还能流畅地"),
]

N_VALUES = [5, 10, 15, 20]


def test_snippet(model, label, text):
    print(f"\n{'=' * 60}")
    print(f"  [{label}]  text = \"{text[:40]}...\"")
    print(f"{'=' * 60}")

    for n in N_VALUES:
        if n > len(text):
            continue

        prefix = text[:n]
        actual_next = text[n] if n < len(text) else "?"

        chars, probs = model.predict(prefix)

        low, mid = all_bounds(probs)

        # Find rank of actual next char
        actual_rank = "?"
        for i, c in enumerate(chars):
            if c == actual_next:
                actual_rank = i + 1
                break

        top5 = list(zip(chars[:5], probs[:5]))

        print(f"\n  N={n:<3}  prefix=\"{prefix}\"")
        print(f"         next=\"{actual_next}\" (rank={actual_rank})  H_low={low:.2f}  H_mid={mid:.2f}  bits")
        print(f"         top5: {[(c, f'{p:.3f}') for c, p in top5]}")


def main():
    print("Loading model...")
    model = CharModel()
    print(f"  Device: {model.device}")
    print(f"  Char vocab: {len(model.char_ids)} tokens")

    for label, text in SNIPPETS:
        test_snippet(model, label, text)

    print("\n\nDone.")


if __name__ == "__main__":
    main()

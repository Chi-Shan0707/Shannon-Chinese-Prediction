"""
Shannon-style Chinese Prediction Experiment — Token-Match Design

Walk through text segments (sentences/paragraphs). For each segment:
  - Feed prefix (growing from warmup to N_max=50 chars)
  - At each step, model predicts ranked tokens
  - Check if any predicted token matches the actual next text (startswith)
  - Record match info, advance by matched token length

Data per record: step, pos, prefix_len, matched_rank/prob/logprob/token/token_len,
                 entropy, top1_prob, top1_token, source, segment_id

Categories: human_jianshi, sushi_qiren, wiki, news, kfc
Output: pj/results/{category}.jsonl

Usage:
  conda activate qwenenv
  python pj/src/run_experiment.py [--max-chars 0] [--top-k 1000] [--n-max 50]
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "pj/models/Qwen3-0.6B"
CLEAN_DIR = Path("pj/data/clean")
OUTPUT_DIR = Path("pj/results")

CATEGORY_FILES = {
    "human_jianshi": ["human_jianshi.txt"],
    "sushi_qiren": ["sushi_qiren.txt"],
    "wiki": ["wiki_samples.txt"],
    "news": [
        "news_01_矿难_bbc.txt",
        "news_02_矿难_新华社调查.txt",
        "news_03_脑机接口.txt",
        "news_04_黄仁勋专访.txt",
        "news_05_西夏陵申遗.txt",
        "news_06_毒杨梅调查.txt",
        "news_07_学术打假.txt",
        "news_08_气候大会.txt",
    ],
    "kfc": ["kfc_original.txt"],
}

WARMUP = 5


def split_segments(text: str) -> list[str]:
    """Split text into segments by double-newline, then single lines."""
    segments = []
    for block in text.split("\n\n"):
        for line in block.strip().split("\n"):
            line = line.strip()
            if len(line) > WARMUP + 3:
                segments.append(line)
    return segments


class ShannonPredictor:
    def __init__(self, top_k: int = 1000, n_max: int = 50):
        print("Loading tokenizer...")
        self.tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float32, device_map="auto"
        )
        self.device = next(self.model.parameters()).device
        self.top_k = top_k
        self.n_max = n_max
        self.vocab_size = self.tok.vocab_size
        print(f"  Device: {self.device}, Vocab: {self.vocab_size}")
        print(f"  Top-K: {self.top_k}, N_max: {self.n_max}, Warmup: {WARMUP}")

    @torch.no_grad()
    def predict(self, prefix: str):
        inputs = self.tok(prefix, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits[0, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)

        topk_logp, topk_ids = torch.topk(log_probs, self.top_k)
        ranked = []
        for rank_0, (tid, lp) in enumerate(zip(topk_ids.tolist(), topk_logp.tolist())):
            token_str = self.tok.decode([tid])
            prob = math.exp(lp)
            ranked.append({
                "rank": rank_0 + 1,
                "token": token_str,
                "logprob": round(lp, 6),
                "prob": round(prob, 8),
            })

        top1_prob = math.exp(topk_logp[0].item())
        entropy = -(probs * log_probs).sum().item()

        return {
            "ranked": ranked,
            "entropy": round(entropy, 6),
            "top1_prob": round(top1_prob, 8),
            "top1_token": ranked[0]["token"],
        }

    def walk_segment(self, text: str, seg_id: int, source: str):
        if len(text) < WARMUP + 1:
            return []

        records = []
        pos = WARMUP
        step = 0

        while pos < len(text) and pos < self.n_max:
            prefix = text[:pos]
            remaining = text[pos:]

            if not remaining:
                break

            pred = self.predict(prefix)
            ranked = pred["ranked"]

            matched_rank = -1
            matched_token = ""
            matched_logprob = 0.0
            matched_prob = 0.0
            matched_len = 0

            for item in ranked:
                if item["token"] and remaining.startswith(item["token"]):
                    matched_rank = item["rank"]
                    matched_token = item["token"]
                    matched_logprob = item["logprob"]
                    matched_prob = item["prob"]
                    matched_len = len(item["token"])
                    break

            if matched_rank == -1 or matched_len == 0:
                advance = 1
            else:
                advance = matched_len

            records.append({
                "step": step,
                "pos": pos,
                "prefix_len": len(prefix),
                "matched_rank": matched_rank,
                "matched_token": matched_token,
                "matched_logprob": matched_logprob,
                "matched_prob": matched_prob,
                "matched_token_len": matched_len,
                "entropy": pred["entropy"],
                "top1_prob": pred["top1_prob"],
                "top1_token": pred["top1_token"],
                "source": source,
                "seg_id": seg_id,
            })

            pos += advance
            step += 1

        return records


def run_category(predictor, category: str, files: list[str], max_chars: int, max_segs: int):
    all_records = []

    for fname in files:
        fpath = CLEAN_DIR / fname
        if not fpath.exists():
            print(f"  SKIP {fname} (not found)")
            continue

        text = fpath.read_text(encoding="utf-8")
        if max_chars > 0:
            text = text[:max_chars]

        segments = split_segments(text)
        if max_segs > 0 and len(segments) > max_segs:
            segments = segments[:max_segs]
        print(f"  {fname}: {len(segments)} segments", flush=True)

        t0 = time.time()
        for seg_id, seg in enumerate(segments):
            records = predictor.walk_segment(seg, seg_id, fname)
            all_records.extend(records)
            if seg_id % 50 == 0 and seg_id > 0:
                elapsed = time.time() - t0
                print(f"    {seg_id}/{len(segments)} segs, {len(all_records)} steps, {elapsed:.1f}s", flush=True)

        elapsed = time.time() - t0
        print(f"    done: {len(segments)} segs → {len(all_records)} steps in {elapsed:.1f}s", flush=True)

    out_path = OUTPUT_DIR / f"{category}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return all_records


def summarize(records, category: str):
    if not records:
        print(f"  [{category}] No records")
        return

    matched = [r for r in records if r["matched_rank"] > 0]
    missed = [r for r in records if r["matched_rank"] == -1]
    rank1 = sum(1 for r in matched if r["matched_rank"] == 1)
    total = len(records)

    avg_rank = sum(r["matched_rank"] for r in matched) / len(matched) if matched else 0
    avg_logprob = sum(r["matched_logprob"] for r in matched) / len(matched) if matched else 0
    avg_prob = sum(r["matched_prob"] for r in matched) / len(matched) if matched else 0
    avg_entropy = sum(r["entropy"] for r in records) / len(records) if records else 0

    print(f"\n  [{category}] {total} steps, {len(set(r['seg_id'] for r in records))} segments")
    print(f"    Matched:   {len(matched)} ({100*len(matched)/total:.1f}%)")
    print(f"    Missed:    {len(missed)} ({100*len(missed)/total:.1f}%)")
    print(f"    Rank-1:    {rank1} ({100*rank1/total:.1f}%)")
    print(f"    Avg rank:  {avg_rank:.2f}")
    print(f"    Avg logp:  {avg_logprob:.4f}")
    print(f"    Avg prob:  {avg_prob:.6f}")
    print(f"    Avg H:     {avg_entropy:.4f} bits")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chars", type=int, default=0,
                        help="Max chars per file (0=all)")
    parser.add_argument("--max-segs", type=int, default=200,
                        help="Max segments per file")
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--n-max", type=int, default=50,
                        help="Max prefix length per segment (chars)")
    parser.add_argument("--categories", nargs="*", default=None)
    args = parser.parse_args()

    predictor = ShannonPredictor(top_k=args.top_k, n_max=args.n_max)
    cats = args.categories or list(CATEGORY_FILES.keys())

    for cat in cats:
        if cat not in CATEGORY_FILES:
            print(f"Unknown category: {cat}")
            continue
        print(f"\n{'='*50}")
        print(f"Category: {cat}")
        print(f"{'='*50}")
        records = run_category(predictor, cat, CATEGORY_FILES[cat], args.max_chars, args.max_segs)
        summarize(records, cat)

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

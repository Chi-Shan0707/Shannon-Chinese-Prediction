"""
KFC Reverse-Context Experiment — minimal/speed version.

For a few selected KFC sentences, pick anchor positions and test
how prediction entropy drops as context grows going backward.
"""

import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = str(BASE / "models" / "Qwen3-0.6B")
DATA_PATH = str(BASE / "data" / "clean" / "kfc_original.txt")
OUTPUT_PATH = str(BASE / "results" / "kfc_reverse_context.jsonl")

CONTEXT_LENS = [5, 10, 20, 50]  # chars before anchor

device = "cpu"


def load_kfc_lines(path):
    return [l.strip() for l in Path(path).read_text(encoding="utf-8").split("\n") if len(l.strip()) > 10]


def find_anchors_in_line(line):
    """Find positions of key phrases in a line."""
    positions = set()
    for kw in ["肯德基", "星期四", "v我50", "V我50", "v50", "V50", "KFC"]:
        idx = 0
        while True:
            idx = line.find(kw, idx)
            if idx < 0:
                break
            positions.add(idx)
            idx += 1
    # also add middle of line
    if len(line) > 15:
        positions.add(len(line) // 2)
    return sorted(p for p in positions if 5 < p < len(line) - 3)


@torch.no_grad()
def predict_prefix(tok, model, prefix, top_k=1000):
    inputs = tok(prefix, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
    logits = model(**inputs).logits[0, -1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum().item()

    topk_logp, topk_ids = torch.topk(log_probs, top_k)
    top_tokens = []
    for i, (tid, lp) in enumerate(zip(topk_ids.tolist(), topk_logp.tolist())):
        top_tokens.append({
            "rank": i + 1,
            "token": tok.decode([tid]),
            "logprob": round(lp, 6),
            "prob": round(math.exp(lp), 8),
        })

    return {
        "entropy": round(entropy, 6),
        "top1_token": top_tokens[0]["token"],
        "top1_prob": top_tokens[0]["prob"],
        "top5": top_tokens[:5],
    }


def main():
    print("Loading...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
    model.to("cpu").eval()
    print("Ready on CPU")

    lines = load_kfc_lines(DATA_PATH)
    all_records = []

    for sent_id, line in enumerate(lines):
        anchors = find_anchors_in_line(line)
        if not anchors:
            continue
        for pos in anchors:
            actual_char = line[pos] if pos < len(line) else ""
            for ctx_len in CONTEXT_LENS:
                prefix_start = max(0, pos - ctx_len)
                prefix = line[prefix_start:pos]

                pred = predict_prefix(tok, model, prefix)
                remaining = line[pos:]

                matched_rank = -1
                matched_token = ""
                for item in pred["top5"]:
                    if item["token"] and remaining.startswith(item["token"]):
                        matched_rank = item["rank"]
                        matched_token = item["token"]
                        break

                all_records.append({
                    "sent_id": sent_id,
                    "pos": pos,
                    "context_len": len(prefix),
                    "actual_char": actual_char,
                    "matched_rank": matched_rank,
                    "matched_token": matched_token,
                    "entropy": pred["entropy"],
                    "top1_token": pred["top1_token"],
                    "top1_prob": pred["top1_prob"],
                    "top5_tokens": [t["token"] for t in pred["top5"]],
                })
        print(f"  Sent {sent_id}: {len(anchors)} anchors", flush=True)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(all_records)} records to {OUTPUT_PATH}")

    print(f"\n{'='*60}")
    print("ENTROPY vs CONTEXT LENGTH (averaged)")
    print(f"{'='*60}")
    for ctx_len in CONTEXT_LENS:
        subset = [r for r in all_records if r["context_len"] == min(ctx_len, max(r["context_len"] for r in all_records if r["context_len"] <= ctx_len))]
    # Actually just group by actual context_len (which may differ from requested)
    from collections import defaultdict
    by_len = defaultdict(list)
    for r in all_records:
        by_len[r["context_len"]].append(r)
    for clen in sorted(by_len):
        subset = by_len[clen]
        avg_entropy = sum(r["entropy"] for r in subset) / len(subset)
        matched = [r for r in subset if r["matched_rank"] > 0]
        match_rate = len(matched) / len(subset) * 100
        rank1 = sum(1 for r in matched if r["matched_rank"] == 1)
        rank1_rate = rank1 / len(subset) * 100 if subset else 0
        print(f"  context={clen:3d}: {len(subset):3d} preds, H={avg_entropy:.4f}, match={match_rate:.1f}%, rank1={rank1_rate:.1f}%")

    print(f"\n{'='*60}")
    print("SAMPLE: Top-5 at '星期四' positions (context=50)")
    print(f"{'='*60}")
    for r in all_records:
        if r["context_len"] >= 30 and "星期四" in r.get("actual_char", ""):
            print(f"  H={r['entropy']:.4f}, top5={r['top5_tokens']}")


if __name__ == "__main__":
    main()

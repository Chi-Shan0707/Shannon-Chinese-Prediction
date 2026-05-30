"""
Targeted twist prediction test.

For each KFC meme, find the punchline position and test:
  - Given prefix up to just before the punchline keyword
  - What does the model predict?
  - Is the actual punchline in the top-K?

Also test internet_twists (反转段子/废话文学/etc.) at their twist points.
"""

import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = str(BASE / "models" / "Qwen3-0.6B")
KFC_PATH = BASE / "data" / "clean" / "kfc_original.txt"
TWISTS_PATH = BASE / "data" / "clean" / "internet_twists.txt"
OUTPUT_PATH = BASE / "results" / "twist_prediction.jsonl"


@torch.no_grad()
def predict_at_point(tok, model, prefix, top_k=1000):
    inputs = tok(prefix, return_tensors="pt").to("cuda:0")
    logits = model(**inputs).logits[0, -1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum().item()

    topk_logp, topk_ids = torch.topk(log_probs, top_k)
    ranked = []
    for i, (tid, lp) in enumerate(zip(topk_ids.tolist(), topk_logp.tolist())):
        ranked.append({
            "rank": i + 1,
            "token": tok.decode([tid]),
            "logprob": round(lp, 6),
            "prob": round(math.exp(lp), 8),
        })
    return {
        "entropy": round(entropy, 6),
        "ranked": ranked,
    }


def find_keyword_rank(ranked, keyword):
    """Find if keyword appears as a prefix of any ranked token."""
    for item in ranked:
        if item["token"] and (keyword.startswith(item["token"]) or item["token"].startswith(keyword)):
            return item
    return None


def main():
    print("Loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16)
    model.to("cuda:0").eval()
    print("Ready on GPU")

    all_records = []

    # === KFC memes: test at punchline positions ===
    print("\n=== KFC Memes ===")
    kfc_lines = [l.strip() for l in KFC_PATH.read_text(encoding="utf-8").split("\n") if len(l.strip()) > 10]

    # Define punchline targets for each KFC meme
    # Format: (prefix_before_punchline, target_keywords, description)
    kfc_tests = []
    for line in kfc_lines:
        # Find where "星期四" or "肯德基" or "v我50" or "V我50" appears
        for kw in ["星期四", "肯德基", "v我50", "V我50", "v50", "V50", "KFC"]:
            idx = line.find(kw)
            if idx > 0:
                prefix = line[:idx]
                kfc_tests.append((prefix, kw, line[:min(len(line), 80)]))
                break  # one test per line

    for prefix, target, context in kfc_tests:
        pred = predict_at_point(tok, model, prefix)
        ranked = pred["ranked"]

        # Search for target in ranked tokens
        match = find_keyword_rank(ranked, target)
        # Also search for individual chars of target
        char_matches = []
        for ch in target:
            for item in ranked[:100]:
                if item["token"].startswith(ch):
                    char_matches.append({"char": ch, "rank": item["rank"], "token": item["token"]})
                    break

        record = {
            "type": "kfc",
            "prefix": prefix[-30:],  # last 30 chars of prefix
            "target": target,
            "target_found": match is not None,
            "target_rank": match["rank"] if match else -1,
            "target_prob": match["prob"] if match else 0.0,
            "target_token": match["token"] if match else "",
            "entropy": pred["entropy"],
            "top5": [(t["token"], t["prob"]) for t in ranked[:5]],
            "context": context,
        }
        all_records.append(record)

        status = f"Rank {match['rank']}" if match else "NOT FOUND"
        print(f"\n  Prefix: ...{prefix[-20:]}")
        print(f"  Target: '{target}' → {status}")
        print(f"  Top-5: {[t[0] for t in record['top5']]}")
        print(f"  Entropy: {pred['entropy']:.4f}")

    # === Internet twists: test at reversal points ===
    print("\n=== Internet Twists ===")
    twists_text = TWISTS_PATH.read_text(encoding="utf-8")
    twists_lines = [l.strip() for l in twists_text.split("\n") if len(l.strip()) > 5]

    # For each twist, find the punchline (usually last few chars/words)
    # and test if model can predict it from the setup
    for line in twists_lines:
        if len(line) < 15:
            continue
        # Use last 1/3 as the "twist", first 2/3 as prefix
        split_point = len(line) * 2 // 3
        # Find a natural word boundary near split_point
        for i in range(split_point, min(split_point + 10, len(line))):
            if line[i] in '，。！？、；：':
                split_point = i + 1
                break
        prefix = line[:split_point]
        remaining = line[split_point:]
        if len(remaining) < 2:
            continue

        pred = predict_at_point(tok, model, prefix)
        ranked = pred["ranked"]

        # Check if any top-K token matches the start of remaining text
        match = None
        for item in ranked:
            if item["token"] and remaining.startswith(item["token"]):
                match = item
                break

        record = {
            "type": "twist",
            "prefix": prefix[-30:],
            "remaining_start": remaining[:20],
            "match_found": match is not None,
            "match_rank": match["rank"] if match else -1,
            "match_prob": match["prob"] if match else 0.0,
            "match_token": match["token"] if match else "",
            "entropy": pred["entropy"],
            "top5": [(t["token"], t["prob"]) for t in ranked[:5]],
            "full_text": line,
        }
        all_records.append(record)

        status = f"Rank {match['rank']}" if match else "NOT FOUND"
        print(f"\n  ...{prefix[-25:]}| ← predict → |{remaining[:15]}...")
        print(f"  Match: {status}, H={pred['entropy']:.4f}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    kfc_records = [r for r in all_records if r["type"] == "kfc"]
    twist_records = [r for r in all_records if r["type"] == "twist"]

    if kfc_records:
        found = sum(1 for r in kfc_records if r["target_found"])
        print(f"\nKFC punchline prediction ({len(kfc_records)} tests):")
        print(f"  Target found in top-1000: {found}/{len(kfc_records)} ({100*found/len(kfc_records):.0f}%)")
        avg_rank = sum(r["target_rank"] for r in kfc_records if r["target_found"]) / max(found, 1)
        print(f"  Avg rank (when found): {avg_rank:.1f}")
        avg_h = sum(r["entropy"] for r in kfc_records) / len(kfc_records)
        print(f"  Avg entropy at punchline: {avg_h:.4f}")

    if twist_records:
        found = sum(1 for r in twist_records if r["match_found"])
        print(f"\nTwist prediction ({len(twist_records)} tests):")
        print(f"  Match found in top-1000: {found}/{len(twist_records)} ({100*found/len(twist_records):.0f}%)")
        avg_rank = sum(r["match_rank"] for r in twist_records if r["match_found"]) / max(found, 1)
        print(f"  Avg rank (when found): {avg_rank:.1f}")
        avg_h = sum(r["entropy"] for r in twist_records) / len(twist_records)
        print(f"  Avg entropy at twist: {avg_h:.4f}")


if __name__ == "__main__":
    main()

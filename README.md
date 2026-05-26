# Shannon-Chinese-Prediction

Replicate Shannon's (1951) letter prediction experiment for **Chinese characters** using a local LLM (Qwen3-0.6B Base).

## Method

Token-match Shannon guessing game:
1. Feed prefix C to model
2. Model returns ranked tokens Y₁, Y₂, Y₃, ... (by probability)
3. Check each Yₖ: does `C + Yₖ` match original text?
4. First match → record rank(k), logprob, advance len(Yₖ) chars
5. No match in top-K → record miss, advance 1 char

No char-to-token mapping needed — tokenizer's native multi-char tokens ("我们", "知道") are handled naturally.

## Categories

| Category | Source | Segments |
|----------|--------|----------|
| human_jianshi | 《人类简史》Chinese edition | 200 |
| sushi_qiren | 《俗世奇人》vernacular fiction | 200 |
| wiki | Wikipedia extracts (120 articles) | 120 |
| news | 8 scraped news articles (2025-2026) | 94 |
| kfc | "疯狂星期四" internet memes | 15 |

## Preliminary Results (N=5→50 chars context)

| Category | Match Rate | Rank-1 | Avg P(match) | Avg H (bits) |
|----------|-----------|--------|-------------|-------------|
| wiki | 94.0% | **27.6%** | 0.204 | **3.53** |
| human_jianshi | 91.6% | 24.7% | 0.181 | 3.95 |
| news | 92.0% | 24.6% | 0.184 | 3.80 |
| kfc | 91.4% | 24.2% | 0.159 | 4.22 |
| sushi_qiren | 90.0% | 19.6% | 0.137 | **4.48** |

Wiki is most predictable (H=3.53 bits/char), vernacular fiction is hardest (H=4.48).

## Requirements

- Python 3.10+
- PyTorch, Transformers
- Qwen3-0.6B (download via ModelScope)
- GPU recommended (tested on RTX 5060 8GB)

```bash
conda activate qwenenv
python src/run_experiment.py --top-k 1000 --n-max 50
```

## Why This Matters

Shannon (1951) measured English entropy via human guessing: H₀=4.03, H∞≈1.3 bits/letter. Chinese — with ~6700 common characters — has never had a prediction experiment (infeasible for humans). Known bounds: H₀≈9.62 (Sun & Sun), neural LM bound≈3.96 bpc (Takahashi & Tanaka-Ishii 2018).

A direct Chinese character conditional entropy chain using LLM prediction.

# Shannon Chinese Prediction Experiment

Replicate Shannon's (1951) prediction experiment for Chinese texts using Qwen3-0.6B Base as an automatic predictor.

## Quick Start

```bash
conda activate qwenenv

# Run experiment (all categories, 200 segments each):
python src/run_experiment.py --top-k 1000 --n-max 50 --max-segs 200

# Quick test (single category):
python src/run_experiment.py --top-k 1000 --n-max 50 --categories kfc

# Generate figures:
python src/make_figures.py
```

## Method

Token-match Shannon guessing game:

1. Feed text prefix to model
2. Model computes a full-vocabulary next-token distribution
3. Check whether any top-1000 token is a prefix of the remaining text
4. First match → record rank, probability, entropy; advance by token length
5. No match in top-K → record miss; advance 1 char

Model: Qwen3-0.6B Base (no RLHF), local inference on RTX 5060 8GB.

## Project Structure

```
report.html              Final HTML report
fig_*.png                Publication-quality figures
src/run_experiment.py    Main experiment runner
src/make_figures.py      Figure generation (matplotlib)
src/clean_data.py        Raw text → cleaned UTF-8
src/prepare_new_data.py  Literature + internet twists data prep
src/kfc_reverse_context.py   KFC twist-point prediction test
src/twist_prediction.py      Internet twist prediction test
data/                    Raw and cleaned text data
data/clean/              Cleaned UTF-8 segments (input to experiment)
results/                 JSONL output (gitignored, local only)
```

## Data (11 Categories)

| Category | Source | Segments |
|----------|--------|----------|
| wiki | Wikipedia samples | 120 |
| news | Scraped news 2025-2026 | 94 |
| human_jianshi | 《人类简史》 | 200 |
| sushi_qiren | 《俗世奇人》 | 200 |
| sanguo | 《三国演义》 | 200 |
| tianlongbabu | 《天龙八部》 | 200 |
| bailuyuan | 《白鹿原》 | 200 |
| sishitongtang | 《四世同堂》 | 200 |
| wanli | 《万历十五年》 | 200 |
| kfc | 疯狂星期四 memes | 42 |
| internet_twists | 反转段子/废话文学 | 63 |

## Results Summary

Entropy values are next-token distribution entropy in nats/step, not character-level bits/char entropy rates.

| Category | Steps | Match Rate | Rank-1 | Avg H (nats/step) |
|----------|-------|------------|--------|-------------------|
| wiki | 4228 | 94.0% | 27.6% | 3.53 |
| news | 2693 | 92.0% | 24.6% | 3.80 |
| internet_twists | 1323 | 92.3% | 25.6% | 3.94 |
| human_jianshi | 6450 | 91.6% | 24.7% | 3.95 |
| kfc | 1497 | 91.4% | 27.5% | 3.95 |
| sanguo | 7818 | 95.0% | 25.7% | 4.00 |
| wanli | 7590 | 91.3% | 23.4% | 4.07 |
| tianlongbabu | 7425 | 92.0% | 24.2% | 4.09 |
| bailuyuan | 7399 | 90.1% | 20.1% | 4.12 |
| sishitongtang | 6188 | 91.0% | 18.1% | 4.38 |
| sushi_qiren | 6885 | 90.0% | 19.6% | 4.48 |

Extended experiment (N=5→200, power-law fit H(N)=H∞+a·N^(-b), nats/step):

| Text | H∞ | b | R² |
|------|-----|---|-----|
| sanguo | 3.00 | 0.681 | 0.99 |
| bailuyuan | 3.22 | 0.541 | 0.99 |
| human_jianshi | 3.21 | 1.035 | 0.93 |
| tianlongbabu | 3.05 | 0.955 | 0.93 |

Do not compute Chinese redundancy by directly dividing these token-level nats/step values by character-level H₀. Strict redundancy requires character-level probability aggregation first.

## Environment

- Conda env: `qwenenv`
- GPU: RTX 5060 8GB, CUDA 13.1
- Model: Qwen3-0.6B (1.4GB, in `models/`, gitignored)
- Python: torch, transformers

# Chinese Character Entropy — Shannon Prediction Experiment

## Goal

Replicate Shannon's (1951) prediction experiment for **Chinese characters** using a local LLM (Qwen3-0.6B Base).

## Core Idea (Shannon Guessing Game for Tokens)

Walk through original text step by step:
1. Feed prefix C (text so far) to model
2. Model returns ranked predicted tokens Y₁, Y₂, Y₃, ... (by probability)
3. Check each Yₖ: does `C + Yₖ` match the original text?
4. First match → record `rank(k)`, `logprob`, `prob`, advance by `len(Yₖ)` chars
5. No match in top-K → record miss, advance 1 char

**Key insight**: Yₖ is a **token** (not a character). It can be 1 char ("的"), 2 chars ("我们"), or more. If rank-1 "我们" matches the next 2 chars of original text → correct, skip 2 chars. This naturally handles tokenizer multi-char tokens with zero char↔token mapping complexity.

**"匹配" definition**: token string is a **prefix** of the remaining text. If rank-1 = "我们" and remaining = "我们的..." → match! Both characters "我" and "们" are correctly predicted in one step.

## Why This Matters

- Shannon (1951) measured English entropy via human guessing: H₀=4.03, H∞≈1.3 bits/letter
- Chinese has never had a Shannon-style prediction experiment (6700-char alphabet makes human experiments infeasible)
- Known Chinese entropy bounds: H₀≈9.62 bits/char (Sun & Sun), bigram→7.15, neural LM→3.96 bpc (Takahashi & Tanaka-Ishii 2018)
- Our experiment attempts to construct a conditional entropy chain H(N) for Chinese

## Methodology

### Model
- **Qwen3-0.6B Base** (not Instruct) — clean LM probabilities, not distorted by RLHF
- Local inference via Transformers, CUDA (RTX 5060 8GB)
- Tokenizer: 151,643 vocab, ~25K tokens start with CJK
- Knowledge cutoff: ~mid-2025

### Experiment Design
- Text is split into segments (paragraphs/lines)
- Each segment: start with warmup=5 chars, predict at every position until N_max=50 chars
- No hard char↔token mapping — use raw tokenizer output and string matching
- Skip non-CJK positions naturally (they just stay in context, never predicted)

### Data Collected Per Step
| Field | Meaning |
|-------|---------|
| step | Prediction step within segment |
| pos | Character offset in original text |
| prefix_len | Length of context fed to model (chars) |
| matched_rank | Rank of matching token (1-based), -1 = no match |
| matched_token | The token string that matched |
| matched_logprob | Log probability of the matched token |
| matched_prob | Probability of the matched token |
| matched_token_len | How many chars this token covers |
| entropy | Full distribution entropy (bits) |
| top1_prob | Probability of model's best guess |
| top1_token | Model's best guess token |
| source | Source file name |
| seg_id | Segment index within file |

## Results (N=5→50, top-K=1000)

### Summary

| Category | Segments | Steps | Match Rate | Rank-1 | Avg Rank | Avg P(match) | Avg H (bits) |
|----------|----------|-------|------------|--------|----------|--------------|--------------|
| human_jianshi | 200 | 6450 | 91.6% | 24.7% | 84.0 | 0.181 | **3.95** |
| sushi_qiren | 200 | 6885 | 90.0% | 19.6% | 103.3 | 0.137 | **4.48** |
| wiki | 120 | 4228 | 94.0% | **27.6%** | **77.1** | 0.204 | **3.53** |
| news | 94 | 2693 | 92.0% | 24.6% | 81.4 | 0.184 | **3.80** |
| kfc | 15 | 549 | 91.4% | 24.2% | 88.2 | 0.159 | **4.22** |

### Key Findings
- **wiki (encyclopedic) is most predictable**: H=3.53 bits/char, Rank-1=27.6% — consistent with formal, terminology-dense text
- **sushi_qiren (vernacular fiction) is hardest**: H=4.48 bits/char, Rank-1=19.6% — literary style, more surprisal
- **human_jianshi (popular non-fiction)**: H=3.95 — in between, expository prose
- **kfc (internet memes)**: H=4.22 — higher entropy than expected, meme culture is creative
- **news**: H=3.80 — journalistic writing is fairly predictable
- All categories achieve ~90-94% match rate within top-1000 predictions

### What These Numbers Mean
- H=3.53 bits/char (wiki) means: on average, each Chinese character carries 3.53 bits of information given ~25 chars of context
- This is consistent with the neural LM lower bound of ~3.96 bpc from literature
- As context grows from N=5 to N=50, entropy should decrease (the conditional entropy chain)
- Our result is a **snapshot at N=5→50**; the full H(N) curve requires running at many N values

## Data Pipeline

### Raw Data
| File | Source | Size | Notes |
|------|--------|------|-------|
| `data/人类简史.txt` | 《人类简史》Chinese edition (GB18030) | 572KB | Full book, TOC + content |
| `data/俗世奇人.txt` | 《俗世奇人》novel | 87KB | With website headers |
| `data/chinese_samples.txt` | Wikipedia samples (120 articles) | 58KB | Random Wikipedia extracts |
| `data/KFC.txt` | "疯狂星期四" memes | 5KB | 15 KFC meme stories (user-added) |
| `data/news_01-08.txt` | Scraped news 2025-2026 | 24KB | 8 articles |

### Cleaning
Script: `src/clean_data.py`
- Removes encoding artifacts (Wikipedia `\1\2`)
- Strips metadata headers/footers, standalone numeric labels
- Preserves digits, punctuation, English-in-Chinese, KFC memes as-is
- All outputs in `data/clean/` as UTF-8

### Categories
- `human_jianshi`: 200 segments (233K chars total)
- `sushi_qiren`: 200 segments (87K chars total)
- `wiki`: 120 segments (58K chars total)
- `news`: 94 segments (24KB total, 8 files)
- `kfc`: 15 segments (5KB)

## Scripts

| Script | Purpose |
|--------|---------|
| `src/run_experiment.py` | Main experiment runner — token-match Shannon guessing game |
| `src/clean_data.py` | Raw text → cleaned UTF-8 segments |
| `src/entropy_calculator.py` | Shannon inequality bounds (unused in current approach) |
| `src/local_model.py` | Original char-level prediction (deprecated) |
| `src/experiment.py` | Original experiment pipeline (deprecated) |
| `src/smoke_test.py` | Quick validation (N=5/10/15/20, deprecated) |

## Experiment Design Evolution

### Phase 1: Char-level (discarded)
- Mask to single-char CJK tokens only (8501/151643)
- Discards probability mass from 16807 multi-char tokens
- Complex char↔token mapping, buggy

### Phase 2: Token aggregation (proposed but skipped)
- P(char c) = Σ P(token) for all tokens starting with c
- Renormalize over CJK chars
- Correct but computationally expensive

### Phase 3 (current): Token-match
- Model predicts tokens directly, check string match against original text
- No char↔token mapping at all
- Multi-char tokens naturally handled (rank-1 "我们" → skip 2 chars)
- Much simpler, cleaner, and uses model's raw output

## Next Steps

1. **KFC-specific test**: predict at "v我50，吃___" position to see model's top predictions
2. **Full H(N) chain**: run experiment at multiple N values (10, 20, 30, 50, 100, 200) to plot the conditional entropy curve
3. **Analysis scripts**: compute and plot H(N) vs N from the collected data
4. **Compare with literature**: plot against Sun & Sun (H₀=9.62) and Takahashi & Tanaka-Ishii (3.96 bpc)
5. **Compare categories**: which text type has the steepest entropy drop?

## Environment

- Conda env: `qwenenv`
- GPU: RTX 5060 8GB, CUDA 13.1
- Model: `Qwen3-0.6B` (1.4GB, in `models/`, gitignored)
- Python: torch, transformers

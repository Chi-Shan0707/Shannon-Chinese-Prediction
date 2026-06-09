# Shannon-Chinese-Prediction

Shannon-style token-match prediction experiment for Chinese texts using a local
Qwen3-0.6B Base model as an automatic predictor.

Main deliverables:

- `report.html`: Chinese course/report version
- `report_en.html`: English personal-website post version

## Method

Token-match guessing game:

1. Feed a text prefix to the model.
2. Compute the next-token probability distribution over the full vocabulary.
3. Search the top-1000 tokens for the first token that matches the remaining text.
4. Record rank, probability, distribution entropy, and advance by the matched token length.
5. If no top-1000 token matches, record a miss and advance one character.

Entropy is computed with natural logarithms in `src/run_experiment.py`, so the
reported values are model next-token distribution entropy in `nats/step`. They
are useful for comparing texts within this experiment, but they are not strict
character-level `bits/char` entropy rates.

## Data

| Category | Source |
|----------|--------|
| wiki | Wikipedia samples |
| news | 8 news articles |
| internet_twists | Internet twists / anti-jokes |
| human_jianshi | 《人类简史》 |
| kfc | 疯狂星期四 memes |
| sanguo | 《三国演义》 |
| wanli | 《万历十五年》 |
| tianlongbabu | 《天龙八部》 |
| bailuyuan | 《白鹿原》 |
| sishitongtang | 《四世同堂》 |
| sushi_qiren | 《俗世奇人》 |

## Results Summary

Main experiment, `N=5→50`, `top_k=1000`:

| Category | Steps | Match Rate | Rank-1 | Avg H (nats/step) |
|----------|-------|------------|--------|-------------------|
| wiki | 4,228 | 94.0% | 27.6% | 3.53 |
| news | 2,693 | 92.0% | 24.6% | 3.80 |
| internet_twists | 1,323 | 92.3% | 25.6% | 3.94 |
| human_jianshi | 6,450 | 91.6% | 24.7% | 3.95 |
| kfc | 1,497 | 91.4% | 27.5% | 3.95 |
| sanguo | 7,818 | 95.0% | 25.7% | 4.00 |
| wanli | 7,590 | 91.3% | 23.4% | 4.07 |
| tianlongbabu | 7,425 | 92.0% | 24.2% | 4.09 |
| bailuyuan | 7,399 | 90.1% | 20.1% | 4.12 |
| sishitongtang | 6,188 | 91.0% | 18.1% | 4.38 |
| sushi_qiren | 6,885 | 90.0% | 19.6% | 4.48 |

Key takeaways:

- Wikipedia and news are most predictable.
- Dialect-rich fiction is least predictable.
- `sanguo` has the highest top-1000 match rate, showing that formulaic style can
  improve candidate-set hits without making the whole probability distribution
  the sharpest.
- KFC meme tests separate fixed collocations such as `疯狂→星期四` from more
  context-dependent punchline prediction such as `吃→肯德基`.

## Reproduce Figures

```bash
python src/make_figures.py
```

The figure script writes `figs/fig_entropy.png`, `figs/fig_scatter.png`,
`figs/fig_rank1.png`, and `figs/fig_hn_decay.png`.

## Run Experiments

Requirements:

- Python 3.10+
- PyTorch
- Transformers
- Local Qwen3-0.6B model under `models/Qwen3-0.6B`
- GPU recommended

```bash
conda activate qwenenv
python src/run_experiment.py --top-k 1000 --n-max 50 --max-segs 200
```

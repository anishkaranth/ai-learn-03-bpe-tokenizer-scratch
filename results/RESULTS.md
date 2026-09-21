# Smoke results — ai-learn-03-bpe-tokenizer-scratch

**Seed:** `42`

## Headline metrics

| Metric | Value |
|--------|------:|
| Merges trained | 40 |
| Vocab size | 61 |
| Char-level tokens (test) | 53 |
| BPE tokens (test) | 25 |
| Compression ratio | 2.120 |
| Round-trip OK | True |
| Runtime (s) | 0.407 |

## Example

- text: `lower newest wider hello world transformer attention`
- decoded: `lower newest wider hello world transformer attention`
- top merges: `[['r', '</w>'], ['l', 'o'], ['i', 'n'], ['w', '</w>'], ['e', 'r</w>'], ['o', 'd'], ['n', 'e'], ['in', 'g']]`

## Plots

- [`vocab_growth.png`](vocab_growth.png) / [`vocab_growth.svg`](vocab_growth.svg)
- [`token_counts.png`](token_counts.png) / [`token_counts.svg`](token_counts.svg)

## Takeaway

BPE repeatedly merges the most frequent adjacent pairs, growing a subword vocab that shortens token sequences on repeated stems (`low`/`lower`, `wide`/`wider`).

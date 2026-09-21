# AI Learn 03 — BPE Tokenizer from Scratch

Train a toy **Byte-Pair Encoding** tokenizer on a tiny corpus: learn merge rules, encode/decode, and plot vocab growth vs compression.

## Learning goals

- Why subword tokenization beats raw characters / whitespace words
- The BPE training loop: count pairs → merge most frequent → repeat
- How vocab size and compression ratio trade off with `#merges`
- Round-trip encode → decode on held-out text

## Layout

```
bpe_tokenizer.py
run_smoke.py
notebooks/bpe_tokenizer.ipynb
results/
```

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_smoke.py
```

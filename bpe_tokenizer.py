"""Byte-Pair Encoding (BPE) tokenizer from scratch (toy educational implementation)."""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

Pair = Tuple[str, str]


def get_stats(sequences: Sequence[Sequence[str]]) -> Counter:
    """Count adjacent symbol pairs across all sequences."""
    stats: Counter = Counter()
    for seq in sequences:
        for i in range(len(seq) - 1):
            stats[(seq[i], seq[i + 1])] += 1
    return stats


def merge_vocab(sequences: List[List[str]], pair: Pair) -> List[List[str]]:
    """Replace every occurrence of `pair` with the joined token."""
    a, b = pair
    merged = a + b
    out: List[List[str]] = []
    for seq in sequences:
        new_seq: List[str] = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                new_seq.append(merged)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        out.append(new_seq)
    return out


class BPETokenizer:
    """Train merges on a tiny corpus; encode / decode with the learned vocab."""

    def __init__(self) -> None:
        self.merges: List[Pair] = []
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    def _word_to_chars(self, word: str) -> List[str]:
        chars = list(word) + ["</w>"]
        return chars

    def train(self, corpus: str, num_merges: int = 50) -> None:
        words = corpus.split()
        sequences = [self._word_to_chars(w) for w in words]
        base_tokens = sorted({t for seq in sequences for t in seq})
        self.merges = []
        for _ in range(num_merges):
            stats = get_stats(sequences)
            if not stats:
                break
            best = max(stats.items(), key=lambda kv: (kv[1], kv[0]))[0]
            if stats[best] < 1:
                break
            sequences = merge_vocab(sequences, best)
            self.merges.append(best)
        tokens = set(base_tokens)
        for a, b in self.merges:
            tokens.add(a + b)
        for seq in sequences:
            tokens.update(seq)
        self.vocab = {t: i for i, t in enumerate(sorted(tokens))}
        self.id_to_token = {i: t for t, i in self.vocab.items()}

    def encode_word(self, word: str) -> List[str]:
        seq = self._word_to_chars(word)
        for pair in self.merges:
            seq = merge_vocab([seq], pair)[0]
        return seq

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for word in text.split():
            for tok in self.encode_word(word):
                if tok not in self.vocab:
                    for ch in tok:
                        ids.append(self.vocab.get(ch, 0))
                else:
                    ids.append(self.vocab[tok])
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        tokens = [self.id_to_token[i] for i in ids]
        return "".join(tokens).replace("</w>", " ").strip()

    def compression_stats(self, text: str) -> Dict[str, float]:
        words = text.split()
        char_tokens = sum(len(w) + 1 for w in words)
        bpe_tokens = len(self.encode(text))
        return {
            "n_words": float(len(words)),
            "char_tokens": float(char_tokens),
            "bpe_tokens": float(bpe_tokens),
            "compression_ratio": float(char_tokens) / max(bpe_tokens, 1),
            "vocab_size": float(len(self.vocab)),
            "n_merges": float(len(self.merges)),
        }

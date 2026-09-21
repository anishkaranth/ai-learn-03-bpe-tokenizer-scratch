#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from bpe_tokenizer import BPETokenizer
CORPUS = 'low low lower lowest newer wider new new hello world transformer attention byte pair encoding'
TEST = 'lower newest wider hello world transformer attention'
RESULTS = Path(__file__).resolve().parent / 'results'
def main():
    tok = BPETokenizer(); tok.train(CORPUS, num_merges=40)
    dec = tok.decode(tok.encode(TEST)); stats = tok.compression_stats(TEST)
    RESULTS.mkdir(exist_ok=True)
    shot = {'snapshot':'smoke_run','seed':42,'vocab_size':int(stats['vocab_size']),'compression_ratio':float(stats['compression_ratio']),'roundtrip_ok':dec.split()==TEST.split(),'runtime_s':0.0}
    (RESULTS/'JSON.shot').write_text(json.dumps(shot, indent=2)+'\n'); print(json.dumps(shot, indent=2))
if __name__ == '__main__':
    main()

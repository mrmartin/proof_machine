"""synth/seed_buffer.py — convert hol_synth_corpus.jsonl into a
microgpt_expit BUFFER_JSONL.

The expit trainer at startup loads its buffer from BUFFER_PATH if the
file exists; otherwise it seeds from the curated ENCODED_CORPUS.  By
pre-writing a buffer file populated with synthetic samples, we
substitute the corpus without modifying the trainer itself.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--include-curated", action="store_true",
                    help="prepend the curated ENCODED_CORPUS entries "
                         "(so the model sees both)")
    args = p.parse_args()

    written = 0
    out = open(args.out, "w")

    if args.include_curated:
        HERE = os.path.dirname(os.path.abspath(__file__))
        ROOT = os.path.abspath(os.path.join(HERE, ".."))
        sys.path.insert(0, ROOT)
        sys.path.insert(0, os.path.join(ROOT, "tokenizer", "python"))
        from microgpt_with_RL_hol import ENCODED_CORPUS
        for goal_toks, cert_toks in ENCODED_CORPUS:
            out.write(json.dumps({
                "goal_toks": list(goal_toks),
                "cert_toks": list(cert_toks),
                "source": "curated",
                "discovered_round": 0,
            }) + "\n")
            written += 1

    with open(args.inp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.write(json.dumps({
                "goal_toks": d["goal_toks"],
                "cert_toks": d["cert_toks"],
                "source": "synthetic",
                "discovered_round": 0,
            }) + "\n")
            written += 1

    out.close()
    print(f"wrote {written} buffer entries to {args.out}")


if __name__ == "__main__":
    main()

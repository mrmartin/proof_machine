"""synth/audit_buffer.py — sample-reverify the existing buffer JSONL.

Before the variant_of injective-renaming fix, the build pipeline
emitted alpha-renamed variants that could collide on the destination
name, producing tokens that look syntactically valid but don't
kernel-verify.  Only the first 50 attempts per build batch were
sanity-checked; the rest landed in the buffer unchecked.

This script samples N random buffer entries and re-runs them through
the OCaml kernel, reporting the contamination rate broken down by
`source` (curated / synthetic / discovered).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tokenizer", "python"))

from kernel_client import KernelClient


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--buffer", default=os.path.join(ROOT, "hol_expit_buffer.jsonl"))
    p.add_argument("--num", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Count total lines so we can sample uniformly.
    print(f"counting lines in {args.buffer}...", flush=True)
    n_total = 0
    with open(args.buffer) as f:
        for _ in f:
            n_total += 1
    print(f"  total: {n_total}")

    rng = random.Random(args.seed)
    sample_indices = set(rng.sample(range(n_total),
                                       min(args.num, n_total)))

    client = KernelClient()
    by_source_total = Counter()
    by_source_reject = Counter()
    rejected_rule_first_token = Counter()
    t0 = time.time()
    n_checked = 0

    with open(args.buffer) as f:
        for idx, line in enumerate(f):
            if idx not in sample_indices:
                continue
            d = json.loads(line)
            source = d.get("source", "?")
            by_source_total[source] += 1
            v = client.verify(d["goal_toks"], d["goal_toks"]) if False else \
                client.verify(d["cert_toks"], d["goal_toks"])
            if v != 100:
                by_source_reject[source] += 1
            n_checked += 1
            if n_checked % 500 == 0:
                elapsed = time.time() - t0
                print(f"  {n_checked}/{len(sample_indices)}  "
                      f"rate={n_checked/elapsed:.0f}/s",
                      flush=True)

    client.close()

    print(f"\n=== Buffer contamination audit, n={n_checked} sampled ===")
    print(f"  buffer file:  {args.buffer}")
    print(f"  total lines:  {n_total}")
    print()
    print(f"  {'source':<14} {'total':>7} {'reject':>7} {'rate':>7}")
    for src in sorted(by_source_total):
        tot = by_source_total[src]
        rej = by_source_reject[src]
        pct = 100.0 * rej / max(1, tot)
        print(f"  {src:<14} {tot:>7d} {rej:>7d} {pct:>6.3f}%")
    total_rej = sum(by_source_reject.values())
    total_n = sum(by_source_total.values())
    print(f"  {'OVERALL':<14} {total_n:>7d} {total_rej:>7d} "
          f"{100.0 * total_rej / max(1, total_n):>6.3f}%")
    print(f"\nelapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

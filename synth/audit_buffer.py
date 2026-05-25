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
import hol_tokenizer as T


def cert_rule_shape(cert_toks):
    """Inline shape extractor (kept here so this script has no PyTorch
    dependency via microgpt_expit)."""
    rules = []
    n = len(cert_toks)
    for i in range(n - 1):
        if cert_toks[i] == T.KW_RULE:
            for j in range(i + 1, min(i + 4, n)):
                t = cert_toks[j]
                if T.RULE_FIRST <= t <= T.RULE_LAST:
                    rules.append(T.RULE_FROM_TOK[t])
                    break
    return tuple(rules)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--buffer", default=os.path.join(ROOT, "hol_expit_buffer.jsonl"))
    p.add_argument("--num", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source", default=None,
                    help="restrict sampling to entries with this source "
                         "(curated/synthetic/discovered)")
    args = p.parse_args()

    # Build a list of (line_no, line) for entries matching the source
    # filter (if any).  We count by scanning the file once, then sample.
    print(f"counting lines in {args.buffer}"
          f"{f' (source={args.source})' if args.source else ''}...",
          flush=True)
    eligible_line_nos = []
    with open(args.buffer) as f:
        for ln, line in enumerate(f):
            if args.source is None:
                eligible_line_nos.append(ln)
            else:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("source", "?") == args.source:
                    eligible_line_nos.append(ln)
    n_total = len(eligible_line_nos)
    print(f"  eligible entries: {n_total}")

    rng = random.Random(args.seed)
    sample_set = set(rng.sample(eligible_line_nos,
                                  min(args.num, n_total)))

    client = KernelClient()
    by_source_total = Counter()
    by_source_reject = Counter()
    # Per-rule: rule_total[r] = certs in sample whose shape contains r;
    #          rule_reject[r] = of those, how many kernel-rejected.
    # Note: a cert with shape (A,B,A,C) contributes once to each of {A,B,C}.
    rule_total = Counter()
    rule_reject = Counter()
    # Rejected-cert shapes (for spotting structural patterns).
    rejected_shapes = Counter()
    t0 = time.time()
    n_checked = 0

    with open(args.buffer) as f:
        for idx, line in enumerate(f):
            if idx not in sample_set:
                continue
            d = json.loads(line)
            source = d.get("source", "?")
            by_source_total[source] += 1
            shape = cert_rule_shape(d["cert_toks"])
            shape_rules = set(shape)
            for r in shape_rules:
                rule_total[r] += 1
            v = client.verify(d["cert_toks"], d["goal_toks"])
            if v != 100:
                by_source_reject[source] += 1
                for r in shape_rules:
                    rule_reject[r] += 1
                rejected_shapes[shape] += 1
            n_checked += 1
            if n_checked % 500 == 0:
                elapsed = time.time() - t0
                print(f"  {n_checked}/{len(sample_set)}  "
                      f"rate={n_checked/elapsed:.0f}/s",
                      flush=True)

    client.close()

    print(f"\n=== Buffer contamination audit, n={n_checked} sampled ===")
    print(f"  buffer file:  {args.buffer}")
    print(f"  source filter: {args.source or '(all)'}")
    print(f"  eligible:     {n_total}")
    print()
    print(f"  {'source':<14} {'total':>7} {'reject':>7} {'rate':>7}")
    for src in sorted(by_source_total):
        tot = by_source_total[src]
        rej = by_source_reject[src]
        pct = 100.0 * rej / max(1, tot)
        print(f"  {src:<14} {tot:>7d} {rej:>7d} {pct:>6.3f}%")
    total_rej = sum(by_source_reject.values())
    total_n = sum(by_source_total.values())
    overall_rate = 100.0 * total_rej / max(1, total_n)
    print(f"  {'OVERALL':<14} {total_n:>7d} {total_rej:>7d} "
          f"{overall_rate:>6.3f}%")

    # Per-rule breakdown.  Sorted by rejection rate desc (worst first),
    # with rare rules surfacing because of low total.
    print()
    print(f"=== Per-rule drift (cert-level, by set-membership in shape) ===")
    print(f"  {'rule':<12} {'certs':>7} {'reject':>7} {'rate':>8} {'vs_overall':>10}")
    rule_rows = []
    for r in rule_total:
        tot = rule_total[r]
        rej = rule_reject[r]
        rate = 100.0 * rej / max(1, tot)
        ratio = rate / max(0.001, overall_rate)
        rule_rows.append((r, tot, rej, rate, ratio))
    # Sort by rejection rate descending, ties by inverse total (rarer first).
    rule_rows.sort(key=lambda x: (-x[3], x[1]))
    for r, tot, rej, rate, ratio in rule_rows:
        marker = "  <<<" if ratio >= 2.0 and rej >= 3 else ""
        print(f"  {r:<12} {tot:>7d} {rej:>7d} {rate:>7.3f}% {ratio:>9.2f}x{marker}")

    # Top rejected shapes (for triage).
    if rejected_shapes:
        print()
        print(f"=== Top 15 rejected rule-shapes ===")
        for shape, cnt in rejected_shapes.most_common(15):
            print(f"  {cnt:>4d}× {' -> '.join(shape) if shape else '(empty)'}")

    print(f"\nelapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

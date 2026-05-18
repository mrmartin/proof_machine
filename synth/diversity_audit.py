"""synth/diversity_audit.py — measure the actual diversity of the
rule-shape distribution in the buffer.

`unique_shape_count` is a misleading metric for a sequence corpus: it
goes up with combinatorial multiplication of position-wise choices,
not with the breadth of choices at any single position.  A corpus
where every shape is `ASSUME · X · X · … · DISCH` produces millions
of "unique shapes" while exposing the model to only one prefix and
one suffix.

This script computes:

  - Position-by-position rule distribution: at slot i (i=0 is the
    first rule applied), what's the histogram over the 17 rules?
  - Per-position entropy (in nats), normalised by log(17).
  - Modal rule at each position and its mass.
  - Common "shape skeleton" fractions: ASSUME-prefix, DISCH-suffix,
    ASSUME…DISCH sandwich.
  - Length distribution.

Output is a one-screen text report plus per-position CSV for plotting.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tokenizer", "python"))

import hol_tokenizer as T
from microgpt_expit import cert_rule_shape


RULE_NAMES = T.RULE_NAMES
N_RULES = len(RULE_NAMES)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--buffer", default=os.path.join(ROOT, "hol_expit_buffer.jsonl"))
    p.add_argument("--source", default=None,
                    help="restrict to entries with this source "
                         "(curated/synthetic/discovered)")
    p.add_argument("--csv", default=None,
                    help="path to write the per-position CSV")
    p.add_argument("--max-pos", type=int, default=12,
                    help="report rule distribution for positions [0..N-1]")
    args = p.parse_args()

    pos_counts = [Counter() for _ in range(args.max_pos)]
    pos_total  = [0] * args.max_pos
    length_hist = Counter()
    unique_shapes = set()
    shapes_by_skeleton = defaultdict(int)
    first_rule_count = Counter()
    last_rule_count  = Counter()
    starts_assume = 0
    ends_disch    = 0
    assume_disch_sandwich = 0   # starts with ASSUME and ends with DISCH
    contains_inst = 0
    contains_spec = 0
    contains_gen  = 0
    contains_mp   = 0
    contains_trans = 0
    contains_mk_comb = 0
    contains_abs = 0
    n = 0
    n_by_source = Counter()

    with open(args.buffer) as f:
        for line in f:
            d = json.loads(line)
            if args.source and d.get("source") != args.source:
                continue
            shape = cert_rule_shape(d["cert_toks"])
            if not shape:
                continue
            n += 1
            n_by_source[d.get("source", "?")] += 1
            unique_shapes.add(shape)
            length_hist[len(shape)] += 1
            first_rule_count[shape[0]] += 1
            last_rule_count[shape[-1]] += 1
            if shape[0] == "ASSUME":
                starts_assume += 1
            if shape[-1] == "DISCH":
                ends_disch += 1
            if shape[0] == "ASSUME" and shape[-1] == "DISCH":
                assume_disch_sandwich += 1
            if "INST" in shape: contains_inst += 1
            if "SPEC" in shape: contains_spec += 1
            if "GEN" in shape:  contains_gen += 1
            if "MP" in shape:   contains_mp += 1
            if "TRANS" in shape: contains_trans += 1
            if "MK_COMB" in shape: contains_mk_comb += 1
            if "ABS" in shape:  contains_abs += 1
            for i, r in enumerate(shape[:args.max_pos]):
                pos_counts[i][r] += 1
                pos_total[i] += 1

    if n == 0:
        print("no entries matched the filter")
        return

    log_n_rules = math.log(N_RULES)

    print(f"=== diversity audit ===")
    print(f"  buffer:        {args.buffer}")
    print(f"  filter:        source={args.source or '(any)'}")
    print(f"  entries:       {n}")
    print(f"    by source:   " + "  ".join(f"{k}={v}" for k, v in n_by_source.most_common()))
    print(f"  unique shapes: {len(unique_shapes)}")
    print()
    print("=== shape length histogram ===")
    for L in sorted(length_hist):
        bar = "#" * int(60 * length_hist[L] / max(length_hist.values()))
        print(f"  len {L:>2}: {length_hist[L]:>7d}  {bar}")
    print()
    print("=== skeleton-pattern fractions ===")
    def pct(x): return 100.0 * x / n
    print(f"  starts with ASSUME:        {starts_assume:>7d}  ({pct(starts_assume):5.1f}%)")
    print(f"  ends with DISCH:           {ends_disch:>7d}  ({pct(ends_disch):5.1f}%)")
    print(f"  ASSUME...DISCH sandwich:   {assume_disch_sandwich:>7d}  ({pct(assume_disch_sandwich):5.1f}%)")
    print(f"  contains INST:             {contains_inst:>7d}  ({pct(contains_inst):5.1f}%)")
    print(f"  contains SPEC:             {contains_spec:>7d}  ({pct(contains_spec):5.1f}%)")
    print(f"  contains GEN:              {contains_gen:>7d}  ({pct(contains_gen):5.1f}%)")
    print(f"  contains MP:               {contains_mp:>7d}  ({pct(contains_mp):5.1f}%)")
    print(f"  contains TRANS:            {contains_trans:>7d}  ({pct(contains_trans):5.1f}%)")
    print(f"  contains MK_COMB:          {contains_mk_comb:>7d}  ({pct(contains_mk_comb):5.1f}%)")
    print(f"  contains ABS:              {contains_abs:>7d}  ({pct(contains_abs):5.1f}%)")
    print()
    print("=== first-rule / last-rule modes ===")
    print("  first rule:")
    for rule, c in first_rule_count.most_common(5):
        print(f"    {rule:<22} {c:>7d}  ({pct(c):5.1f}%)")
    print("  last rule:")
    for rule, c in last_rule_count.most_common(5):
        print(f"    {rule:<22} {c:>7d}  ({pct(c):5.1f}%)")
    print()
    print("=== per-position rule distribution ===")
    print(f"  {'pos':>3}  {'N':>7}  {'H/Hmax':>7}  {'modal rule':<22} {'modal frac':>10}  {'top 3 rules':<60}")
    for i in range(args.max_pos):
        tot = pos_total[i]
        if tot == 0:
            print(f"  {i:>3}  {tot:>7d}  --       --                       --")
            continue
        # Entropy normalised by log(N_RULES).
        H = 0.0
        for c in pos_counts[i].values():
            p_ = c / tot
            if p_ > 0:
                H -= p_ * math.log(p_)
        H_norm = H / log_n_rules
        modal_rule, modal_count = pos_counts[i].most_common(1)[0]
        top3 = pos_counts[i].most_common(3)
        top3_s = "  ".join(f"{r}({100*c/tot:.0f}%)" for r, c in top3)
        print(f"  {i:>3}  {tot:>7d}  {H_norm:>6.3f}   {modal_rule:<22} "
              f"{100*modal_count/tot:>9.1f}%  {top3_s}")

    # CSV dump for plotting.
    if args.csv:
        with open(args.csv, "w", newline="") as fout:
            w = csv.writer(fout)
            w.writerow(["position"] + RULE_NAMES)
            for i in range(args.max_pos):
                row = [i]
                for r in RULE_NAMES:
                    row.append(pos_counts[i].get(r, 0))
                w.writerow(row)
        print(f"\nwrote per-position CSV to {args.csv}")


if __name__ == "__main__":
    main()

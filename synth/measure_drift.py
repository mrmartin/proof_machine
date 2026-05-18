"""synth/measure_drift.py — measure the shadow→kernel rejection rate.

For each generated proof:
  - The shadow accepts it (otherwise the sample wouldn't exist).
  - The OCaml kernel either accepts (100) or rejects (-1) or accepts
    with wrong concl (0).

We want the rejection rate broken down by:
  - Overall (shadow-emitted that kernel rejects)
  - By final-step rule (which rules' shadow implementations are
    most often wrong)
  - By rule-shape complexity (does drift correlate with depth?)
  - Wrong-concl rate separately (verdict 0 means kernel ran the proof
    but concluded something other than the cert's declared concl — a
    different kind of bug: shadow's apply_step produces a different
    theorem than the kernel's).

Usage:
  python3 synth/measure_drift.py --num 5000

This is purely diagnostic.  It writes nothing to the corpus.
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
sys.path.insert(0, HERE)

from kernel_client import KernelClient
from synth.backward_gen import generate_one, encode_pair, variant_of


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=5000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-depth", type=int, default=2)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--variants", type=int, default=1,
                    help="alpha-renamed variants per skeleton (1 = no variants)")
    args = p.parse_args()

    rng = random.Random(args.seed)
    client = KernelClient()

    n_shadow_emitted = 0
    n_encode_failed  = 0
    n_kernel_accept  = 0   # verdict == 100
    n_kernel_wrong_concl = 0  # verdict == 0
    n_kernel_reject  = 0   # verdict == -1
    by_final_rule_reject = Counter()
    by_final_rule_total  = Counter()
    by_depth_reject      = Counter()
    by_depth_total       = Counter()
    rejected_shapes      = Counter()
    t0 = time.time()

    while n_shadow_emitted < args.num:
        depth = rng.randint(args.min_depth, args.max_depth)
        gr = generate_one(rng, depth)
        if gr is None:
            continue

        # Test the skeleton + (args.variants - 1) alpha-renamed copies.
        proofs_to_check = [gr]
        for _ in range(args.variants - 1):
            v = variant_of(gr, rng)
            if v is not None:
                proofs_to_check.append(v)

        for vr in proofs_to_check:
            n_shadow_emitted += 1
            try:
                cert_toks, goal_toks, c_hdr, g_hdr = encode_pair(vr)
            except Exception:
                n_encode_failed += 1
                continue

            final_rule = vr.steps[-1].rule
            proof_depth = len(vr.steps)
            by_final_rule_total[final_rule] += 1
            by_depth_total[proof_depth] += 1

            v = client.verify(cert_toks, goal_toks,
                               cert_names=list(c_hdr.names),
                               goal_names=list(g_hdr.names))
            if v == 100:
                n_kernel_accept += 1
            elif v == 0:
                n_kernel_wrong_concl += 1
                by_final_rule_reject[final_rule] += 1
                by_depth_reject[proof_depth] += 1
                rejected_shapes[vr.rule_shape] += 1
            else:
                n_kernel_reject += 1
                by_final_rule_reject[final_rule] += 1
                by_depth_reject[proof_depth] += 1
                rejected_shapes[vr.rule_shape] += 1
            if n_shadow_emitted >= args.num:
                break

        if n_shadow_emitted % 500 == 0:
            elapsed = time.time() - t0
            rej_pct = 100.0 * (n_kernel_reject + n_kernel_wrong_concl) / max(1, n_shadow_emitted)
            print(f"  {n_shadow_emitted}/{args.num}  "
                  f"reject={rej_pct:.2f}%  "
                  f"rate={n_shadow_emitted/elapsed:.0f}/s",
                  flush=True)

    client.close()

    print(f"\n=== Shadow→kernel drift, n={n_shadow_emitted} samples ===")
    print(f"  kernel accept (100):       {n_kernel_accept:>5d}  "
          f"({100.0 * n_kernel_accept / n_shadow_emitted:.3f}%)")
    print(f"  kernel wrong concl (0):    {n_kernel_wrong_concl:>5d}  "
          f"({100.0 * n_kernel_wrong_concl / n_shadow_emitted:.3f}%)")
    print(f"  kernel reject (-1):        {n_kernel_reject:>5d}  "
          f"({100.0 * n_kernel_reject / n_shadow_emitted:.3f}%)")
    print(f"  encode failed:             {n_encode_failed:>5d}")
    print(f"  total drift rate:          {100.0 * (n_kernel_reject + n_kernel_wrong_concl) / n_shadow_emitted:.3f}%")
    print()
    print("=== Reject rate by final-step rule ===")
    print(f"  {'rule':<22} {'reject':>7} {'total':>7} {'rate':>6}")
    for rule in sorted(by_final_rule_total, key=lambda r: -by_final_rule_total[r]):
        tot = by_final_rule_total[rule]
        rej = by_final_rule_reject[rule]
        pct = 100.0 * rej / max(1, tot)
        print(f"  {rule:<22} {rej:>7d} {tot:>7d} {pct:>5.2f}%")
    print()
    print("=== Reject rate by proof depth ===")
    print(f"  {'depth':>5} {'reject':>7} {'total':>7} {'rate':>6}")
    for d in sorted(by_depth_total):
        tot = by_depth_total[d]
        rej = by_depth_reject[d]
        pct = 100.0 * rej / max(1, tot)
        print(f"  {d:>5d} {rej:>7d} {tot:>7d} {pct:>5.2f}%")
    if rejected_shapes:
        print()
        print("=== Top 10 rejected rule shapes ===")
        for shape, count in rejected_shapes.most_common(10):
            print(f"  {count:>4d}x  {' -> '.join(shape)}")
    print(f"\nelapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

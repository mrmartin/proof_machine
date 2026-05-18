"""synth/build_corpus.py — generate a synthetic corpus of (goal, cert)
pairs via the shadow + kernel re-verify pipeline, and emit it as a
JSONL file consumable by the ExitIt trainer.

Output: each line is a JSON object with keys
  goal_toks: List[int]
  cert_toks: List[int]
  rule_shape: List[str]
  cert_names: List[str]   # NAME pool entries (constants)
  goal_names: List[str]   # NAME pool entries for the goal-side encoding
  source: "synthetic"
  depth: int

Filters:
  - Token-length cap (default 320, matches BLOCK_SIZE)
  - Deduplication by (goal_toks, cert_toks)
  - Optional rule-shape diversity cap

Usage:
  python3 synth/build_corpus.py --num 10000 --out hol_synth_corpus.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tokenizer", "python"))
sys.path.insert(0, HERE)

import hol_tokenizer as T
from kernel_client import KernelClient
from synth.backward_gen import generate_one, encode_pair, variant_of


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=10000)
    p.add_argument("--out", default=os.path.join(ROOT, "hol_synth_corpus.jsonl"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--block-size", type=int, default=320)
    p.add_argument("--min-depth", type=int, default=2)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--reverify", action="store_true",
                    help="kernel-reverify every sample (default off "
                         "after the first batch sanity check)")
    p.add_argument("--shape-cap", type=int, default=0,
                    help="max instances per unique rule shape "
                         "(0 = no cap)")
    p.add_argument("--variants-per-skeleton", type=int, default=10,
                    help="for each generated proof, emit this many "
                         "alpha-renamed variants (boosts per-shape "
                         "sample count without re-running rule walk)")
    args = p.parse_args()

    rng = random.Random(args.seed)
    client = KernelClient() if args.reverify else None

    out_f = open(args.out, "w")
    written = 0
    rejected_long = 0
    rejected_kernel = 0
    rejected_dup = 0
    shape_caps = {}
    seen_keys = set()
    unique_shapes = set()
    t0 = time.time()

    # Initial sanity: kernel-reverify first 100 samples to catch bugs.
    sanity_client = client or KernelClient()

    next_log = 1000
    target = args.num
    attempts = 0
    while written < target:
        attempts += 1
        if attempts > target * 100:
            print("WARN: too many attempts, stopping")
            break

        depth = rng.randint(args.min_depth, args.max_depth)
        gr = generate_one(rng, depth)
        if gr is None:
            continue

        # Emit the original + alpha-renamed variants of the same proof.
        # This keeps the rule-shape distribution similar but multiplies
        # the per-shape sample count, which the model needs in order to
        # generalise.
        variants = [gr]
        for _ in range(args.variants_per_skeleton - 1):
            v = variant_of(gr, rng)
            if v is not None:
                variants.append(v)

        for vr in variants:
            try:
                cert_toks, goal_toks, c_hdr, g_hdr = encode_pair(vr)
            except Exception:
                continue

            if len(cert_toks) + len(goal_toks) + 3 > args.block_size:
                rejected_long += 1
                continue

            key = (tuple(goal_toks), tuple(cert_toks))
            if key in seen_keys:
                rejected_dup += 1
                continue

            if args.shape_cap > 0:
                cnt = shape_caps.get(vr.rule_shape, 0)
                if cnt >= args.shape_cap:
                    continue

            if attempts <= 50 or args.reverify:
                v_int = sanity_client.verify(cert_toks, goal_toks,
                                              cert_names=list(c_hdr.names),
                                              goal_names=list(g_hdr.names))
                if v_int != 100:
                    rejected_kernel += 1
                    continue

            seen_keys.add(key)
            shape_caps[vr.rule_shape] = shape_caps.get(vr.rule_shape, 0) + 1
            unique_shapes.add(vr.rule_shape)
            out_f.write(json.dumps({
                "goal_toks": goal_toks,
                "cert_toks": cert_toks,
                "rule_shape": list(vr.rule_shape),
                "cert_names": list(c_hdr.names),
                "goal_names": list(g_hdr.names),
                "source": "synthetic",
                "depth": len(vr.steps),
            }) + "\n")
            written += 1

            if written >= next_log:
                elapsed = time.time() - t0
                print(f"  {written}/{target}  unique_shapes={len(unique_shapes)}  "
                      f"len_rejects={rejected_long}  dup_rejects={rejected_dup}  "
                      f"kernel_rejects={rejected_kernel}  "
                      f"rate={written/elapsed:.0f}/s")
                next_log += 5000
            if written >= target:
                break

    out_f.close()
    if client is None:
        sanity_client.close()
    else:
        client.close()

    print(f"\nwrote {written} samples -> {args.out}")
    print(f"unique rule shapes: {len(unique_shapes)}")
    print(f"rejected (length): {rejected_long}")
    print(f"rejected (dup):    {rejected_dup}")
    print(f"rejected (kernel): {rejected_kernel}")
    print(f"elapsed:           {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

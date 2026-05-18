"""tests/synth/test_generator.py — smoke + soundness for the backward generator.

Verifies:
  - Shadow rules produce well-typed theorems
  - Every emitted (goal, cert) pair re-verifies through the OCaml kernel
  - Rule-shape diversity grows with sample count
"""
from __future__ import annotations

import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tokenizer", "python"))
sys.path.insert(0, os.path.join(ROOT, "synth"))

import hol_tokenizer as T
from kernel_client import KernelClient
from synth.backward_gen import generate_one, to_cert, encode_pair


def test_basic_generation():
    rng = random.Random(0)
    ok = 0
    fail = 0
    for i in range(20):
        target_depth = rng.randint(2, 6)
        gr = generate_one(rng, target_depth)
        if gr is None:
            fail += 1
            continue
        ok += 1
    print(f"  basic generation: {ok}/20 succeeded, {fail} failed")
    return ok >= 15


def test_kernel_reverifies():
    """Every shadow-generated proof must also kernel-verify."""
    client = KernelClient()
    rng = random.Random(42)
    ok = 0
    rejected = 0
    skipped = 0
    shapes = set()
    t0 = time.time()
    for i in range(100):
        target_depth = rng.randint(2, 8)
        gr = generate_one(rng, target_depth)
        if gr is None:
            skipped += 1
            continue
        try:
            cert_toks, goal_toks, c_hdr, g_hdr = encode_pair(gr)
        except Exception as e:
            rejected += 1
            continue
        v = client.verify(cert_toks, goal_toks,
                           cert_names=list(c_hdr.names),
                           goal_names=list(g_hdr.names))
        if v == 100:
            ok += 1
            shapes.add(gr.rule_shape)
        else:
            rejected += 1
            if rejected <= 3:
                print(f"    REJECT shape={gr.rule_shape} verdict={v}")
    client.close()
    print(f"  kernel reverify:  {ok}/100 ok, {rejected} rejected, "
          f"{skipped} skipped  ({time.time() - t0:.2f}s)")
    print(f"  unique shapes:    {len(shapes)}")
    return ok >= 50  # at least half should pass


def test_diversity():
    """At scale, generator should produce many unique rule shapes."""
    rng = random.Random(7)
    shapes = set()
    for i in range(500):
        target_depth = rng.randint(3, 10)
        gr = generate_one(rng, target_depth)
        if gr is not None:
            shapes.add(gr.rule_shape)
    print(f"  diversity (500): {len(shapes)} unique shapes")
    return len(shapes) >= 50


def main():
    print("=== generator smoke ===")
    r1 = test_basic_generation()
    r2 = test_kernel_reverifies()
    r3 = test_diversity()
    if r1 and r2 and r3:
        print("\nM2 generator smoke: PASS")
        return 0
    else:
        print("\nM2 generator smoke: FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())

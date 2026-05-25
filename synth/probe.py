"""synth/probe.py — joint-reachability probe runner.

Enables `HOL_SYNTH_PROBE=1` and drives `backward_gen.generate_one` N
times under the production inverse-frequency sampler.  For each walk,
captures the full per-step record (applicable set, precondition flags
for the four stuck shapes, sampled rule, apply success, RuleError
failure reasons) into a JSONL.

This is a measurement-only tool.  It does not touch the live buffer or
the checkpoint; the trainer can run concurrently.

Usage:
    python3 synth/probe.py --num 200000 --seed 1
    python3 synth/probe.py --num 1000 --out /tmp/probe_smoke.jsonl
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

# Enable probe BEFORE importing backward_gen so the module-load gate
# picks it up.  (It's currently only read for diagnostic logging — the
# actual gating in generate_one is `probe_record is not None` — but we
# set it for symmetry and so the import is clearly probe-mode.)
os.environ["HOL_SYNTH_PROBE"] = "1"

import synth.backward_gen as bg  # noqa: E402
from synth.freq_counter import build_from_buffer  # noqa: E402


DEFAULT_BUFFER = os.path.join(ROOT, "hol_expit_buffer.jsonl")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=200000,
                    help="number of walks to run (default 200000)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--buffer", default=DEFAULT_BUFFER,
                    help="buffer JSONL to snapshot frequency from (default "
                         "hol_expit_buffer.jsonl)")
    p.add_argument("--out", default=None,
                    help="output JSONL path (default "
                         "runs/probe_<UTC-timestamp>.jsonl)")
    p.add_argument("--temp", type=float, default=2.0,
                    help="HOL_EXPIT_SYNTH_TEMP equivalent (default 2.0)")
    p.add_argument("--min-depth", type=int, default=2)
    p.add_argument("--max-depth", type=int, default=10)
    args = p.parse_args()

    if args.out is None:
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        args.out = os.path.join(ROOT, "runs", f"probe_{ts}.jsonl")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    assert bg._PROBE_ENABLED, "probe module imported with flag off"
    print(f"probe: num={args.num}  seed={args.seed}  temp={args.temp}",
          flush=True)
    print(f"  buffer: {args.buffer}", flush=True)
    print(f"  out:    {args.out}", flush=True)

    # Snapshot the inverse-frequency landscape.  This is exactly what
    # `expand_corpus_synthetic` in microgpt_expit.py does each round,
    # so the probe reflects production sampling behaviour rather than a
    # uniform prior.
    t0 = time.time()
    print(f"  building freq snapshot from buffer...", flush=True)
    freq = build_from_buffer(args.buffer)
    print(f"  freq: {freq.total_proofs} proofs ({time.time()-t0:.1f}s)",
          flush=True)

    rng = random.Random(args.seed)
    state = bg.GeneratorState(freq=freq, temperature=args.temp)

    t0 = time.time()
    n_complete = 0
    n_failed = 0
    with open(args.out, "w") as f:
        for walk_id in range(args.num):
            depth = rng.randint(args.min_depth, args.max_depth)
            probe_record = {
                "walk_id": walk_id,
                "target_depth": depth,
                "steps": [],
            }
            gr = bg.generate_one(rng, depth, state=state,
                                  probe_record=probe_record)
            if gr is None:
                n_failed += 1
            else:
                n_complete += 1
            f.write(json.dumps(probe_record) + "\n")
            if (walk_id + 1) % 5000 == 0:
                elapsed = time.time() - t0
                print(f"  {walk_id+1}/{args.num}  "
                      f"rate={(walk_id+1)/elapsed:.0f}/s  "
                      f"complete={n_complete}  failed={n_failed}",
                      flush=True)

    elapsed = time.time() - t0
    print(f"\ndone: {args.num} walks in {elapsed:.1f}s "
          f"({args.num/elapsed:.0f}/s)", flush=True)
    print(f"  complete: {n_complete}", flush=True)
    print(f"  failed:   {n_failed}", flush=True)
    print(f"  output:   {args.out}", flush=True)


if __name__ == "__main__":
    main()

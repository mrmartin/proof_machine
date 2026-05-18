"""synth/clean_buffer.py — drop kernel-invalid entries from the buffer.

The variant renamer had a collision bug (fixed) that produced a small
fraction of synthetic samples whose tokens look like a cert but don't
actually kernel-verify.  This script rewrites the buffer JSONL with
only the entries that the kernel accepts.

The original buffer is backed up to <path>.preclean before rewriting.

Threaded over the verifier pool: ~6k entries/sec on this machine.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import local

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tokenizer", "python"))

from kernel_client import KernelClient


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--buffer", default=os.path.join(ROOT, "hol_expit_buffer.jsonl"))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--backup-suffix", default=".preclean")
    args = p.parse_args()

    backup = args.buffer + args.backup_suffix
    if os.path.exists(backup):
        print(f"WARN: backup already exists at {backup}; skipping rewrite")
        sys.exit(1)
    print(f"backing up {args.buffer} -> {backup}")
    shutil.copy2(args.buffer, backup)

    tls = local()

    def get_client():
        if not hasattr(tls, "c"):
            tls.c = KernelClient()
        return tls.c

    def verify_line(line: str):
        d = json.loads(line)
        c = get_client()
        v = c.verify(d["cert_toks"], d["goal_toks"])
        return (v == 100, line)

    n_total = 0
    n_kept = 0
    n_dropped_by_source = {}
    t0 = time.time()
    out_path = args.buffer + ".clean"
    out_f = open(out_path, "w")
    with open(args.buffer) as f, \
         ThreadPoolExecutor(max_workers=args.threads) as ex:
        # Stream lines through the pool in chunks so memory stays bounded.
        chunk = []
        chunk_size = 1024
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                for ok, ln in ex.map(verify_line, chunk):
                    n_total += 1
                    if ok:
                        out_f.write(ln)
                        n_kept += 1
                    else:
                        src = json.loads(ln).get("source", "?")
                        n_dropped_by_source[src] = n_dropped_by_source.get(src, 0) + 1
                chunk = []
                if n_total % 10000 == 0:
                    print(f"  {n_total}: kept {n_kept}, "
                          f"dropped {n_total - n_kept}, "
                          f"rate={n_total/(time.time()-t0):.0f}/s",
                          flush=True)
        if chunk:
            for ok, ln in ex.map(verify_line, chunk):
                n_total += 1
                if ok:
                    out_f.write(ln)
                    n_kept += 1
                else:
                    src = json.loads(ln).get("source", "?")
                    n_dropped_by_source[src] = n_dropped_by_source.get(src, 0) + 1
    out_f.flush()
    os.fsync(out_f.fileno())
    out_f.close()
    os.replace(out_path, args.buffer)

    print(f"\n=== cleanup done ===")
    print(f"  total checked: {n_total}")
    print(f"  kept:          {n_kept}")
    print(f"  dropped:       {n_total - n_kept}")
    for src, n in sorted(n_dropped_by_source.items()):
        print(f"    {src}: {n}")
    print(f"  elapsed:       {time.time() - t0:.1f}s")
    print(f"  backup:        {backup}")


if __name__ == "__main__":
    main()

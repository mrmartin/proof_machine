"""eval_ood.py — M1 evaluation harness.

Runs flat sampling (1A) and best-first tree search (1B) with several
ablation configs against the 23-seed test set, using the existing
ExitIt checkpoint.

Output: a results table (CSV + stdout) of (seed, method, solved,
kernel_calls, forward_passes, depth, elapsed, novel_shape).

Usage:

    python3 eval_ood.py                          # default sweep
    python3 eval_ood.py --methods 1A 1B-b 1B-f  # subset

Environment:
    HOL_EXPIT_CKPT_PATH — checkpoint to load (default: hol_expit_ckpt.pt)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tokenizer", "python"))
sys.path.insert(0, HERE)

import torch                                # noqa: E402

import hol_tokenizer as T                   # noqa: E402
from kernel_client import KernelClient      # noqa: E402
from microgpt_with_RL_hol_gpu import HOLGPT  # noqa: E402
# Expit trains at block_size=320 (HOL_EXPIT_BLOCK_SIZE default).  We must
# build the model with the same wpe size to load its checkpoint.
BLOCK_SIZE = int(os.environ.get("HOL_EXPIT_BLOCK_SIZE", "320"))
from microgpt_with_RL_hol import (          # noqa: E402
    SEEDS, gold_cert_for,
)
from microgpt_expit import CORPUS_RULE_SHAPES, cert_rule_shape  # noqa: E402

from tree_search_infer import (             # noqa: E402
    search_best_first, search_flat_sample, SearchResult,
)


def load_checkpoint(model: HOLGPT, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return int(ckpt.get("round", 0))


def encode_seed(label: str, goal):
    cert = gold_cert_for(label, goal)
    goal_toks, g_hdr = T.encode_term_only(goal)
    cert_toks, c_hdr = T.encode_cert(cert)
    return {
        "label": label,
        "goal": goal,
        "goal_toks": list(goal_toks),
        "cert_toks": list(cert_toks),
        "cert_names": list(c_hdr.names),
        "goal_names": list(g_hdr.names),
        "gold_steps": len(cert.steps),
        "gold_shape": cert_rule_shape(list(cert_toks)),
    }


# Method registry: name -> kwargs for search.
METHODS = {
    # 1A: flat sampling baseline.
    "1A-T0.7":  dict(kind="flat", num_samples=64, temperature=0.7),
    "1A-T1.0":  dict(kind="flat", num_samples=64, temperature=1.0),
    "1A-T1.2":  dict(kind="flat", num_samples=64, temperature=1.2),
    # 1B: best-first tree search.
    "1B-a":     dict(kind="bf", k_outer=3, b_inner=4, inner_temp=0.9,
                     max_depth=12, max_kernel_calls=500,
                     alpha=1.0, uniform_mix=0.0),
    "1B-b":     dict(kind="bf", k_outer=5, b_inner=4, inner_temp=0.9,
                     max_depth=12, max_kernel_calls=1000,
                     alpha=1.0, uniform_mix=0.0),
    "1B-c":     dict(kind="bf", k_outer=5, b_inner=8, inner_temp=1.0,
                     max_depth=12, max_kernel_calls=1500,
                     alpha=1.0, uniform_mix=0.0),
    "1B-d":     dict(kind="bf", k_outer=8, b_inner=4, inner_temp=0.9,
                     max_depth=12, max_kernel_calls=1500,
                     alpha=1.0, uniform_mix=0.0),
    "1B-e":     dict(kind="bf", k_outer=5, b_inner=4, inner_temp=0.9,
                     max_depth=20, max_kernel_calls=2000,
                     alpha=1.0, uniform_mix=0.0),
    "1B-f":     dict(kind="bf", k_outer=5, b_inner=4, inner_temp=0.9,
                     max_depth=12, max_kernel_calls=1500,
                     alpha=1.0, uniform_mix=0.10),
    "1B-g":     dict(kind="bf", k_outer=5, b_inner=4, inner_temp=0.9,
                     max_depth=12, max_kernel_calls=1000,
                     alpha=0.5, uniform_mix=0.0),
}


def run_method(model, client, seed, method_name, device, ckpt_round) -> dict:
    cfg = METHODS[method_name]
    prompt = [T.BOS] + seed["goal_toks"] + [T.BOS]
    if cfg["kind"] == "flat":
        res = search_flat_sample(
            model, client, prompt, seed["goal_toks"],
            seed["cert_names"], seed["goal_names"], device,
            num_samples=cfg["num_samples"],
            temperature=cfg["temperature"],
            max_depth=20,
            known_corpus_shapes=CORPUS_RULE_SHAPES,
        )
    elif cfg["kind"] == "bf":
        res = search_best_first(
            model, client, prompt, seed["goal_toks"],
            seed["cert_names"], seed["goal_names"], device,
            k_outer=cfg["k_outer"],
            b_inner=cfg.get("b_inner", 4),
            inner_temp=cfg.get("inner_temp", 0.9),
            max_depth=cfg["max_depth"],
            max_kernel_calls=cfg["max_kernel_calls"],
            alpha=cfg["alpha"],
            uniform_mix=cfg["uniform_mix"],
            known_corpus_shapes=CORPUS_RULE_SHAPES,
        )
    else:
        raise ValueError(f"unknown kind: {cfg['kind']}")
    return {
        "seed": seed["label"],
        "method": method_name,
        "solved": int(res.solved),
        "kernel_calls": res.kernel_calls,
        "forward_passes": res.forward_passes,
        "depth": res.depth_reached,
        "elapsed": round(res.elapsed, 3),
        "novel_shape": int(res.novel_shape),
        "rule_shape": "-".join(res.rule_shape),
        "ckpt_round": ckpt_round,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=list(METHODS.keys()))
    parser.add_argument("--seeds", nargs="+", default=None,
                        help="subset of seed labels to evaluate")
    parser.add_argument("--ckpt", default=None,
                        help="checkpoint path (default: $HOL_EXPIT_CKPT_PATH "
                             "or hol_expit_ckpt.pt)")
    parser.add_argument("--out", default=os.path.join(HERE, "runs", "eval_ood.csv"))
    parser.add_argument("--quick", action="store_true",
                        help="run small ablation set for fast turnaround")
    args = parser.parse_args()

    if args.quick:
        args.methods = ["1A-T1.0", "1B-b", "1B-f"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HOLGPT(T.VOCAB_SIZE, BLOCK_SIZE).to(device)
    model.eval()

    ckpt_path = args.ckpt or os.environ.get(
        "HOL_EXPIT_CKPT_PATH",
        os.path.join(HERE, "hol_expit_ckpt.pt"))
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint missing: {ckpt_path}")
        sys.exit(1)
    ckpt_round = load_checkpoint(model, ckpt_path, device)
    print(f"loaded checkpoint from {ckpt_path} (round {ckpt_round})")

    encoded_seeds = [encode_seed(label, goal) for (label, goal, _depth) in SEEDS]
    if args.seeds:
        keep = set(args.seeds)
        encoded_seeds = [s for s in encoded_seeds if s["label"] in keep]

    client = KernelClient()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows: List[dict] = []
    t_total = time.time()
    for method in args.methods:
        t_method = time.time()
        solved = 0
        for seed in encoded_seeds:
            row = run_method(model, client, seed, method, device, ckpt_round)
            rows.append(row)
            solved += row["solved"]
            tag = "OK " if row["solved"] else "   "
            novel_tag = " NOVEL" if row["novel_shape"] and row["solved"] else ""
            print(f"  {method:<10} {seed['label']:<22} {tag}  "
                  f"K={row['kernel_calls']:<5} F={row['forward_passes']:<4} "
                  f"d={row['depth']:<2} t={row['elapsed']:>5.2f}s"
                  f"{novel_tag}")
        n = len(encoded_seeds)
        print(f"  [{method}] solved={solved}/{n}  ({time.time()-t_method:.1f}s)")

    print(f"\ntotal: {time.time()-t_total:.1f}s")
    print(f"writing {len(rows)} rows -> {args.out}")
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary by method.
    print("\n=== Summary ===")
    by_method: Dict[str, Dict[str, int]] = {}
    for r in rows:
        m = r["method"]
        b = by_method.setdefault(m, {"solved": 0, "total": 0, "ood_solved": 0,
                                     "ood_total": 0, "novel": 0})
        b["solved"] += r["solved"]
        b["total"] += 1
        # OOD = the 10 OOD seeds.  Bucket by label.
        OOD = {"disj_imp_self", "conj_of_eqs", "abs_impl", "spec_twice",
               "double_gen_imp", "eq_trans_impl", "mk_comb_impl", "comp_imp",
               "conj_assoc", "triple_conj_intro"}
        if r["seed"] in OOD:
            b["ood_total"] += 1
            b["ood_solved"] += r["solved"]
        if r["novel_shape"] and r["solved"]:
            b["novel"] += 1
    for m, b in by_method.items():
        print(f"  {m:<10}  total={b['solved']:>2}/{b['total']:<2}  "
              f"OOD={b['ood_solved']:>2}/{b['ood_total']:<2}  "
              f"novel_shapes={b['novel']}")

    client.close()


if __name__ == "__main__":
    main()

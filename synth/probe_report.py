"""synth/probe_report.py — read probe JSONL, produce reachability report.

Five numbers per target shape, plus a per-shape Mode A / Mode B / both
interpretation:

  1. Unigram reach rate — walks where the rule entered _applicable at
     least once.  Sanity-check the audit's incidence numbers.
  2. Positioned-precondition reach rate — walks reaching a state where
     the rule was applicable AND the shape's hypothesis-context
     requirement was met.  Headline number: the hypothesis predicts
     this is near zero while (1) is large.
  3. Sample-given-reachable rate — at the (walk,step) pairs where the
     precondition was met and the rule was in `applicable`, fraction
     where _sample_rule actually picked it.  Tests whether inverse-
     frequency steers away from in-context rare rules.
  4. _try_apply success rate when sampled in-context, broken down by
     failure reason.  Mode-A / Mode-B discriminator.
  5. Observed shape count — completed walks whose rule_shape matches
     the gold shape exactly.  Expected 0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tokenizer", "python"))


TARGETS = {
    "mk_comb_impl":      {
        "gold_shape": ("ASSUME","ASSUME","MK_COMB","DISCH","DISCH"),
        "rule":       "MK_COMB",
        "precond":    "mk_comb_impl_ready",
    },
    "double_gen_imp":    {
        "gold_shape": ("ASSUME","GEN","GEN","DISCH"),
        "rule":       "GEN",
        "precond":    "double_gen_imp_ready",
    },
    "conj_assoc":        {
        "gold_shape": ("ASSUME","CONJUNCT1","CONJUNCT2","CONJUNCT1",
                       "CONJUNCT2","CONJ","CONJ","DISCH"),
        "rule":       "CONJ",
        "precond":    "conj_assoc_ready",
    },
    "triple_conj_intro": {
        "gold_shape": ("ASSUME","ASSUME","ASSUME","CONJ","CONJ",
                       "DISCH","DISCH","DISCH"),
        "rule":       "CONJ",
        "precond":    "triple_conj_intro_ready",
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", required=True,
                    help="probe JSONL produced by synth/probe.py")
    args = p.parse_args()

    n_walks = 0
    n_complete = 0
    # Per-target accumulators.
    unigram_reach     = {t: 0 for t in TARGETS}  # walks where rule ∈ applicable ≥1×
    precond_reach     = {t: 0 for t in TARGETS}  # walks where precond.t_ready true ≥1×
    in_context_steps  = {t: 0 for t in TARGETS}  # (walk,step) where precond AND rule∈applicable
    in_context_sampled  = {t: 0 for t in TARGETS}  # of above, where sampled == target rule
    in_context_applied  = {t: 0 for t in TARGETS}  # of sampled, apply_ok true
    in_context_failures = {t: Counter() for t in TARGETS}  # reason -> count
    shape_hits        = {t: 0 for t in TARGETS}

    # Sanity: rule unigram reach distribution.
    all_unigram_reach = Counter()

    with open(args.probe) as f:
        for line in f:
            d = json.loads(line)
            n_walks += 1
            shape = d.get("completed_shape")
            if shape is not None:
                n_complete += 1
                shape_t = tuple(shape)
                for t, cfg in TARGETS.items():
                    if shape_t == cfg["gold_shape"]:
                        shape_hits[t] += 1
            # Per-walk reach flags.
            walk_reach    = {t: False for t in TARGETS}
            walk_precond  = {t: False for t in TARGETS}
            walk_unigrams = set()
            for step in d.get("steps", []):
                applicable = set(step.get("applicable") or [])
                walk_unigrams |= applicable
                precond = step.get("precond") or {}
                for t, cfg in TARGETS.items():
                    if cfg["rule"] in applicable:
                        walk_reach[t] = True
                    if precond.get(cfg["precond"], False):
                        walk_precond[t] = True
                        if cfg["rule"] in applicable:
                            in_context_steps[t] += 1
                            if step.get("sampled") == cfg["rule"]:
                                in_context_sampled[t] += 1
                                if step.get("apply_ok"):
                                    in_context_applied[t] += 1
                                else:
                                    for fr in step.get("failures") or []:
                                        rname, reason = fr
                                        if rname == cfg["rule"]:
                                            in_context_failures[t][reason] += 1
            for r in walk_unigrams:
                all_unigram_reach[r] += 1
            for t in TARGETS:
                if walk_reach[t]:
                    unigram_reach[t] += 1
                if walk_precond[t]:
                    precond_reach[t] += 1

    print(f"=== Probe reachability report ===")
    print(f"  input:           {args.probe}")
    print(f"  walks:           {n_walks}")
    print(f"  complete:        {n_complete} ({100.0*n_complete/max(1,n_walks):.1f}%)")
    print()

    print(f"=== Headline table ===")
    print()
    cols = ("seed", "rule", "unigram_reach", "precond_reach",
            "sample|ctx", "apply_ok|ctx", "shape_hits")
    print(f"  {'seed':<20} {'rule':<10} "
          f"{'unigram':>9} {'precond':>9} "
          f"{'sample|ctx':>11} {'apply|ctx':>10} "
          f"{'shape_hits':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*9} {'-'*9} {'-'*11} {'-'*10} {'-'*10}")
    for t, cfg in TARGETS.items():
        ur = 100.0 * unigram_reach[t] / max(1, n_walks)
        pr = 100.0 * precond_reach[t] / max(1, n_walks)
        ictx = in_context_steps[t]
        sgr = (100.0 * in_context_sampled[t] / ictx) if ictx else float("nan")
        sx  = in_context_sampled[t]
        agr = (100.0 * in_context_applied[t] / sx) if sx else float("nan")
        print(f"  {t:<20} {cfg['rule']:<10} "
              f"{ur:>8.2f}% {pr:>8.2f}% "
              f"{sgr:>10.2f}% {agr:>9.2f}% "
              f"{shape_hits[t]:>10d}")
    print()
    print("  Legend:")
    print("    unigram     = % walks where the rule entered _applicable ≥ 1×")
    print("    precond     = % walks where shape-specific precondition was met ≥ 1×")
    print("                  (rule applicable AND hypothesis-context requirement met)")
    print("    sample|ctx  = of (walk,step) pairs with precond met, % where the")
    print("                  inverse-freq sampler picked the target rule")
    print("    apply|ctx   = of times the rule was sampled in-context, % where")
    print("                  _try_apply succeeded")
    print("    shape_hits  = walks whose completed rule shape matches the gold")
    print()

    # Per-seed failure breakdown (for in-context apply failures).
    print(f"=== In-context _try_apply failures (Mode B signal) ===")
    for t, cfg in TARGETS.items():
        ictx_sampled = in_context_sampled[t]
        n_fail = ictx_sampled - in_context_applied[t]
        if ictx_sampled == 0:
            print(f"  {t}: rule was never sampled in-context "
                  f"(precond never met or rule never picked); no Mode B data")
            continue
        if n_fail == 0:
            print(f"  {t}: 0 failures (apply succeeded every time the rule was "
                  f"sampled in-context)")
            continue
        total = sum(in_context_failures[t].values())
        print(f"  {t}: {n_fail}/{ictx_sampled} sampled-in-context applications "
              f"failed; failure-reason breakdown ({total} total reason events, "
              f"may exceed n_fail if rule's retry loop logs multiple per call):")
        for reason, cnt in in_context_failures[t].most_common(10):
            pct = 100.0 * cnt / total
            print(f"    {cnt:>5d} ({pct:>5.1f}%)  {reason}")
    print()

    # Mode A / Mode B reads.
    print(f"=== Per-shape reads ===")
    for t, cfg in TARGETS.items():
        ur = 100.0 * unigram_reach[t] / max(1, n_walks)
        pr = 100.0 * precond_reach[t] / max(1, n_walks)
        ictx_sampled = in_context_sampled[t]
        applied_pct = ((100.0 * in_context_applied[t] / ictx_sampled)
                       if ictx_sampled else None)
        # Heuristic verdict:
        #   - if precond reach is << unigram reach (e.g. <10% of it) → Mode A
        #   - if precond reach is healthy but applied_pct << 100 → Mode B
        precond_ratio = pr / ur if ur > 0 else 0.0
        starvation = precond_ratio < 0.10
        rejection  = (applied_pct is not None) and applied_pct < 50.0
        if starvation and rejection:
            verdict = "Both"
        elif starvation:
            verdict = "Mode A (applicability starvation)"
        elif rejection:
            verdict = "Mode B (retry-loop rejection)"
        elif pr < 1.0:
            verdict = "Mode A (low absolute precond reach)"
        else:
            verdict = "Neither dominant — examine top failure reasons"
        print(f"\n  {t}  [{verdict}]")
        print(f"    unigram reach: {ur:.2f}%   precond reach: {pr:.2f}%   "
              f"precond/unigram ratio: {precond_ratio:.3f}")
        if ictx_sampled > 0:
            print(f"    sampled in-context: {ictx_sampled} times, "
                  f"applied successfully: {applied_pct:.2f}%")
        else:
            print(f"    rule was never sampled in-context")


if __name__ == "__main__":
    main()

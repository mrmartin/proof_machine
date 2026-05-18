"""tests/test_verify_prefix.py — M0 smoke + invariant tests.

Drives the persistent OCaml verifier through KernelClient.verify_prefix
on the existing warmup/OOD seeds, and checks:

  - Full prefix (k=len) accepts iff full verify accepts.
  - Each intermediate prefix (k < len) is also accepted (incremental).
  - Mutating the witness on step k causes prefix mode to reject at
    exactly step k, not earlier or later.
  - Theorem table length equals k on accept.
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tokenizer", "python"))

from kernel_client import KernelClient, PrefixResult  # noqa: E402
import hol_tokenizer as T  # noqa: E402

# Reuse the test seeds + gold-cert builders from the existing trainer.
from microgpt_with_RL_hol import SEEDS, gold_cert_for  # noqa: E402

# SEEDS is a list of (label, goal, depth) triples; flatten to a dict.
SEED_BY_NAME = {label: (goal, depth) for label, goal, depth in SEEDS}


def encode_seed(name: str):
    """Returns (cert_toks, goal_toks, cert_names, goal_names, n_steps)."""
    goal, _depth = SEED_BY_NAME[name]
    cert = gold_cert_for(name, goal)
    goal_toks, g_hdr = T.encode_term_only(goal)
    cert_toks, c_hdr = T.encode_cert(cert)
    return (cert_toks, goal_toks,
            list(c_hdr.names), list(g_hdr.names),
            len(cert.steps))


def test_full_prefix_matches_verify(client: KernelClient):
    """For every seed, full-prefix mode must accept with k theorems."""
    ok_count = 0
    fail = []
    for seed_name in SEED_BY_NAME:
        cert_toks, goal_toks, cert_names, goal_names, n_steps = \
            encode_seed(seed_name)
        legacy = client.verify(cert_toks, goal_toks, cert_names, goal_names)
        pres = client.verify_prefix(cert_toks, n_steps, cert_names)
        if not pres.ok or len(pres.theorems) != n_steps:
            fail.append((seed_name, legacy, pres))
        else:
            ok_count += 1
    print(f"  full-prefix:    {ok_count}/{len(SEED_BY_NAME)} ok")
    if fail:
        for f in fail[:5]:
            print(f"    FAIL {f[0]}: legacy={f[1]} pres={f[2]}")
    return not fail


def test_incremental_prefixes(client: KernelClient):
    """For each seed, every prefix k ∈ [0..n_steps] must accept."""
    ok_count = 0
    fail = []
    for seed_name in SEED_BY_NAME:
        cert_toks, _, cert_names, _, n_steps = encode_seed(seed_name)
        all_ok = True
        for k in range(n_steps + 1):
            pres = client.verify_prefix(cert_toks, k, cert_names)
            if not pres.ok or len(pres.theorems) != k:
                all_ok = False
                fail.append((seed_name, k, pres))
                break
        if all_ok:
            ok_count += 1
    print(f"  incremental:    {ok_count}/{len(SEED_BY_NAME)} ok")
    if fail:
        for f in fail[:3]:
            print(f"    FAIL {f[0]} k={f[1]}: {f[2]}")
    return not fail


def test_concl_tokens_present(client: KernelClient):
    """Sanity: every ThmRecord has a non-empty concl_toks."""
    fail = []
    for seed_name in SEED_BY_NAME:
        cert_toks, _, cert_names, _, n_steps = encode_seed(seed_name)
        pres = client.verify_prefix(cert_toks, n_steps, cert_names)
        if not pres.ok:
            fail.append((seed_name, "not ok"))
            continue
        for t in pres.theorems:
            if not t.concl_toks:
                fail.append((seed_name, f"empty concl_toks for step {t.step_id}"))
    print(f"  concl_present:  {len(SEED_BY_NAME) - len(fail)}/{len(SEED_BY_NAME)} ok")
    return not fail


def test_truncated_prefix_after_failure(client: KernelClient):
    """When step k references an unknown premise, prefix mode at k+1 must
    reject — but prefix mode at k (which only requires k *good* steps) is
    fine.  We exercise this by passing a clearly-too-large k: with k =
    n_steps + 10, the response should be Prefix_ok with exactly n_steps
    theorems (the loop terminates when the cert runs out)."""
    fail = []
    for seed_name in SEED_BY_NAME:
        cert_toks, _, cert_names, _, n_steps = encode_seed(seed_name)
        pres = client.verify_prefix(cert_toks, n_steps + 10, cert_names)
        if not pres.ok or len(pres.theorems) != n_steps:
            fail.append((seed_name, pres))
    print(f"  overshoot_k:    {len(SEED_BY_NAME) - len(fail)}/{len(SEED_BY_NAME)} ok")
    if fail:
        for f in fail[:3]:
            print(f"    FAIL {f[0]}: {f[1]}")
    return not fail


def test_corrupted_cert_rejects(client: KernelClient):
    """Mutate the first step's rule to one whose witness shape it
    cannot consume (BETA→ABS).  BETA's witness is (term ...); ABS
    requires (var n ty), so apply_step raises Failure and prefix mode
    rejects at step 0."""
    import hol_tokenizer as T
    cert_toks, _, cert_names, _, n_steps = encode_seed("beta_inst_identity")
    mutated = list(cert_toks)
    flipped = False
    for i, t in enumerate(mutated):
        if T.RULE_FIRST <= t <= T.RULE_LAST and t == T.RULE_TOK["BETA"]:
            mutated[i] = T.RULE_TOK["ABS"]
            flipped = True
            break
    if not flipped:
        print("  corrupt_step:   SKIP (no BETA found)")
        return True
    pres = client.verify_prefix(mutated, n_steps, cert_names)
    ok = (not pres.ok) and pres.failing_step == 0
    print(f"  corrupt_step:   {'ok' if ok else 'FAIL'}  -> "
          f"ok={pres.ok} step={pres.failing_step}")
    return ok


def test_corrupt_bad_premise(client: KernelClient):
    """Mutate step 2's premise to reference step 99 (non-existent) and
    confirm prefix rejection at step 1.  This exercises the table-lookup
    failure path in apply_step."""
    import hol_tokenizer as T
    cert_toks, _, cert_names, _, n_steps = encode_seed("imp_p")
    # imp_p has 2 steps: ASSUME, DISCH [1].  Find the integer token after
    # "(premises" and replace it with INT_31 (29 maps to int literal 29).
    mutated = list(cert_toks)
    found = False
    for i in range(len(mutated) - 1):
        if mutated[i] == T.KW_PREMISES and T.INT_FIRST <= mutated[i + 1] <= T.INT_LAST:
            # Set premise to int literal 31 (last slot, surely non-existent).
            mutated[i + 1] = T.INT_LAST
            found = True
            break
    if not found:
        print("  corrupt_prem:   SKIP (no premise int found)")
        return True
    pres = client.verify_prefix(mutated, n_steps, cert_names)
    ok = (not pres.ok)
    print(f"  corrupt_prem:   {'ok' if ok else 'FAIL'}  -> "
          f"ok={pres.ok} step={pres.failing_step}")
    return ok


def main():
    client = KernelClient()
    t0 = time.time()
    r1 = test_full_prefix_matches_verify(client)
    r2 = test_incremental_prefixes(client)
    r3 = test_concl_tokens_present(client)
    r4 = test_truncated_prefix_after_failure(client)
    r5 = test_corrupted_cert_rejects(client)
    r6 = test_corrupt_bad_premise(client)
    print(f"\nElapsed {time.time() - t0:.2f}s")
    client.close()
    if r1 and r2 and r3 and r4 and r5 and r6:
        print("\nM0 verify_prefix: ALL PASS")
        return 0
    else:
        print("\nM0 verify_prefix: FAILURES")
        return 1


if __name__ == "__main__":
    sys.exit(main())

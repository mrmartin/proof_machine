"""synth/backward_gen.py — forward-random-walk synthetic generator.

Constructs random kernel-valid proofs by applying random rules forward
from random base theorems, then uses each constructed theorem's
conclusion as the goal and the construction sequence as the gold proof.

Despite the name "backward generator" (the model later solves the
inverse problem: given a goal, recover the proof), the *construction*
is forward — start from axioms/REFLs/ASSUMEs, apply rules, observe
what theorem appears.

For each sample:
  1. Seed the pool with 1-3 random ASSUME/REFL applications on simple
     random terms.
  2. Sample a target depth uniformly from [2, max_depth].
  3. Loop: pick a random rule weighted by applicability; pick random
     arguments (premises from the pool, witnesses from the term
     sampler); apply via the shadow; on success append to the pool.
  4. Pick the deepest derived theorem (highest step ID) as the goal;
     its construction is the gold cert.

The shadow guarantees consistency but is not trusted — every emitted
proof is re-verified by the OCaml kernel before entering the training
corpus.
"""
from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tokenizer", "python"))
sys.path.insert(0, HERE)

import hol_tokenizer as T
import shadow as S
from freq_counter import PositionRuleFreq


# ---------------------------------------------------------------------------
# Term sampling
# ---------------------------------------------------------------------------
BOOL_TY  = T.bool_ty()
NAT_TY   = T.nat_ty()
ATOM_TYPES = [BOOL_TY, NAT_TY]
VAR_POOL_BOOL = ["p", "q", "r", "s"]
VAR_POOL_NAT  = ["a", "b", "c", "n", "m", "x", "y", "z"]


def sample_atom_term(rng: random.Random, ty=None) -> tuple:
    """Sample a single Var of a random atomic type, or matching `ty`."""
    if ty is None:
        ty = rng.choice(ATOM_TYPES)
    pool = VAR_POOL_BOOL if T.type_equal(ty, BOOL_TY) else VAR_POOL_NAT
    name = rng.choice(pool)
    return T.mk_var(name, ty)


def sample_bool_term(rng: random.Random, depth: int = 0) -> tuple:
    """Random bool-typed term, optionally compound."""
    if depth >= 2 or rng.random() < 0.5:
        return sample_atom_term(rng, BOOL_TY)
    kind = rng.choice(["eq_nat", "imp", "conj", "disj", "forall"])
    if kind == "eq_nat":
        return T.mk_eq(sample_atom_term(rng, NAT_TY),
                       sample_atom_term(rng, NAT_TY))
    if kind == "imp":
        return T.mk_imp(sample_bool_term(rng, depth + 1),
                        sample_bool_term(rng, depth + 1))
    if kind == "conj":
        return T.mk_conj(sample_bool_term(rng, depth + 1),
                         sample_bool_term(rng, depth + 1))
    if kind == "disj":
        return T.mk_disj(sample_bool_term(rng, depth + 1),
                         sample_bool_term(rng, depth + 1))
    if kind == "forall":
        ty = rng.choice(ATOM_TYPES)
        nm = rng.choice(VAR_POOL_NAT if T.type_equal(ty, NAT_TY) else VAR_POOL_BOOL)
        return T.mk_forall(nm, ty, sample_bool_term(rng, depth + 1))
    return sample_atom_term(rng, BOOL_TY)


def sample_nat_term(rng: random.Random) -> tuple:
    return sample_atom_term(rng, NAT_TY)


# ---------------------------------------------------------------------------
# Proof builder
# ---------------------------------------------------------------------------
@dataclass
class ProofStep:
    id: int
    rule: str
    witness: tuple
    premises: List[int]
    thm: S.Thm   # derived theorem (kept for chaining)


@dataclass
class GenResult:
    goal: tuple                  # conclusion of the picked target theorem
    steps: List[ProofStep]      # the chain
    rule_shape: Tuple[str, ...]


def _step_thm(steps: List[ProofStep], id_: int) -> S.Thm:
    for s in steps:
        if s.id == id_:
            return s.thm
    raise IndexError(id_)


def _premise_with(steps: List[ProofStep],
                   pred,
                   rng: random.Random) -> Optional[int]:
    cands = [s.id for s in steps if pred(s.thm)]
    if not cands:
        return None
    return rng.choice(cands)


def _try_apply(rule_name: str,
               steps: List[ProofStep],
               rng: random.Random) -> Optional[ProofStep]:
    """Try one application of `rule_name`.  Returns a new ProofStep on
    success, or None if the rule isn't currently applicable (no valid
    premises, type mismatch, etc.).
    """
    next_id = (steps[-1].id + 1) if steps else 1
    rule_fn, arity, wkind = S.RULES[rule_name]

    # --- pick witness + premises by rule ---
    if rule_name == "REFL":
        t = (sample_bool_term(rng) if rng.random() < 0.5 else sample_nat_term(rng))
        witness = ("term", t)
        try:
            thm = S.refl(t)
        except S.RuleError:
            return None
        return ProofStep(next_id, "REFL", witness, [], thm)

    if rule_name == "ASSUME":
        t = sample_bool_term(rng)
        witness = ("term", t)
        try:
            thm = S.assume(t)
        except S.RuleError:
            return None
        return ProofStep(next_id, "ASSUME", witness, [], thm)

    if rule_name == "BETA":
        # Build a redex (λv. body) v where v occurs in body's type slot.
        ty = rng.choice(ATOM_TYPES)
        name = rng.choice(VAR_POOL_NAT if T.type_equal(ty, NAT_TY) else VAR_POOL_BOOL)
        var = T.mk_var(name, ty)
        body = var  # simplest beta-redex: (λv. v) v → v = v
        if rng.random() < 0.3:
            # (λv. const) v -- another simple form
            body = sample_atom_term(rng, ty)
            if not isinstance(body, tuple) or body[0] != "Var" or body[1] == name:
                body = T.mk_var(name, ty)  # fall back
        redex = T.mk_comb(("Abs", name, ty, body), var)
        try:
            thm = S.beta(redex)
        except S.RuleError:
            return None
        return ProofStep(next_id, "BETA", ("term", redex), [], thm)

    if rule_name == "TRANS":
        # Need two equalities th1: a=b, th2: b=c.  Hardest part: chain.
        eq_steps = [s for s in steps
                    if (s.thm.concl and s.thm.concl[0] == "Comb"
                         and s.thm.concl[1][0] == "Comb"
                         and s.thm.concl[1][1][0] == "Const"
                         and s.thm.concl[1][1][1] == "=")]
        if len(eq_steps) < 2:
            return None
        # Try a few random pairs.
        for _ in range(5):
            a, b = rng.sample(eq_steps, 2)
            try:
                thm = S.trans(a.thm, b.thm)
                return ProofStep(next_id, "TRANS", ("none",),
                                 [a.id, b.id], thm)
            except S.RuleError:
                continue
        return None

    if rule_name == "MK_COMB":
        # Two equalities of compatible types.
        eq_steps = [s for s in steps
                    if (s.thm.concl[0] == "Comb"
                         and s.thm.concl[1][0] == "Comb"
                         and s.thm.concl[1][1][0] == "Const"
                         and s.thm.concl[1][1][1] == "=")]
        if len(eq_steps) < 2:
            return None
        for _ in range(5):
            a, b = rng.sample(eq_steps, 2)
            try:
                thm = S.mk_comb(a.thm, b.thm)
                return ProofStep(next_id, "MK_COMB", ("none",),
                                 [a.id, b.id], thm)
            except S.RuleError:
                continue
        return None

    if rule_name == "ABS":
        # An equality with a fresh var name we can abstract over.
        eq_steps = [s for s in steps
                    if (s.thm.concl[0] == "Comb"
                         and s.thm.concl[1][0] == "Comb"
                         and s.thm.concl[1][1][0] == "Const"
                         and s.thm.concl[1][1][1] == "=")]
        if not eq_steps:
            return None
        for _ in range(5):
            a = rng.choice(eq_steps)
            # Pick a var either appearing free in the concl or fresh.
            fvs = S.free_vars(a.thm.concl)
            if fvs and rng.random() < 0.7:
                name, ty = rng.choice(fvs)
            else:
                ty = rng.choice(ATOM_TYPES)
                pool = (VAR_POOL_NAT if T.type_equal(ty, NAT_TY)
                        else VAR_POOL_BOOL)
                name = rng.choice(pool)
            v = T.mk_var(name, ty)
            try:
                thm = S.abs_rule(v, a.thm)
                return ProofStep(next_id, "ABS", ("var", name, ty),
                                 [a.id], thm)
            except S.RuleError:
                continue
        return None

    if rule_name == "DISCH":
        if not steps:
            return None
        for _ in range(5):
            tgt = rng.choice(steps)
            if not tgt.thm.hyps and rng.random() < 0.6:
                # disch a fresh assumption too
                p = sample_bool_term(rng)
            else:
                if tgt.thm.hyps:
                    p = rng.choice(tgt.thm.hyps)
                else:
                    p = sample_bool_term(rng)
            try:
                thm = S.disch(p, tgt.thm)
                return ProofStep(next_id, "DISCH", ("term", p),
                                 [tgt.id], thm)
            except S.RuleError:
                continue
        return None

    if rule_name == "MP":
        # Need a |- a==>b and a |- a.
        imp_steps = [s for s in steps
                     if (s.thm.concl[0] == "Comb"
                          and s.thm.concl[1][0] == "Comb"
                          and s.thm.concl[1][1][0] == "Const"
                          and s.thm.concl[1][1][1] == "==>")]
        if not imp_steps:
            return None
        for _ in range(5):
            a = rng.choice(imp_steps)
            ante = a.thm.concl[1][2]
            # Find a step proving `ante`.
            matches = [s for s in steps if S.alpha_eq(s.thm.concl, ante)]
            if not matches:
                continue
            b = rng.choice(matches)
            try:
                thm = S.mp(a.thm, b.thm)
                return ProofStep(next_id, "MP", ("none",),
                                 [a.id, b.id], thm)
            except S.RuleError:
                continue
        return None

    if rule_name == "GEN":
        if not steps:
            return None
        for _ in range(5):
            a = rng.choice(steps)
            fvs_concl = S.free_vars(a.thm.concl)
            if not fvs_concl:
                continue
            name, ty = rng.choice(fvs_concl)
            # GEN requires the var not appear free in hyps.
            if any(S.has_free(name, ty, h) for h in a.thm.hyps):
                continue
            v = T.mk_var(name, ty)
            try:
                thm = S.gen(v, a.thm)
                return ProofStep(next_id, "GEN", ("var", name, ty),
                                 [a.id], thm)
            except S.RuleError:
                continue
        return None

    if rule_name == "SPEC":
        # Need a |- ∀v. p
        fa_steps = [s for s in steps
                    if (s.thm.concl[0] == "Comb"
                         and s.thm.concl[1][0] == "Const"
                         and s.thm.concl[1][1] == "!")]
        if not fa_steps:
            return None
        for _ in range(5):
            a = rng.choice(fa_steps)
            # Bound type is the type of the abstraction's bound var.
            abs_ = a.thm.concl[2]
            bound_ty = abs_[2]
            # Sample a term of that type — for simplicity, a var.
            pool = (VAR_POOL_NAT if T.type_equal(bound_ty, NAT_TY)
                    else (VAR_POOL_BOOL if T.type_equal(bound_ty, BOOL_TY)
                          else ["x", "y", "z"]))
            w = T.mk_var(rng.choice(pool), bound_ty)
            try:
                thm = S.spec(w, a.thm)
                return ProofStep(next_id, "SPEC", ("term", w),
                                 [a.id], thm)
            except S.RuleError:
                continue
        return None

    if rule_name == "CONJ":
        if len(steps) < 2:
            return None
        # Pick any two distinct premises whose conclusions are bool.
        bool_steps = [s for s in steps if S.is_bool(s.thm.concl)]
        if len(bool_steps) < 2:
            return None
        for _ in range(5):
            a, b = rng.sample(bool_steps, 2)
            try:
                thm = S.conj(a.thm, b.thm)
                return ProofStep(next_id, "CONJ", ("none",),
                                 [a.id, b.id], thm)
            except S.RuleError:
                continue
        return None

    if rule_name == "CONJUNCT1":
        conj_steps = [s for s in steps
                      if (s.thm.concl[0] == "Comb"
                           and s.thm.concl[1][0] == "Comb"
                           and s.thm.concl[1][1][0] == "Const"
                           and s.thm.concl[1][1][1] == "/\\")]
        if not conj_steps:
            return None
        a = rng.choice(conj_steps)
        try:
            thm = S.conjunct1(a.thm)
            return ProofStep(next_id, "CONJUNCT1", ("none",), [a.id], thm)
        except S.RuleError:
            return None

    if rule_name == "CONJUNCT2":
        conj_steps = [s for s in steps
                      if (s.thm.concl[0] == "Comb"
                           and s.thm.concl[1][0] == "Comb"
                           and s.thm.concl[1][1][0] == "Const"
                           and s.thm.concl[1][1][1] == "/\\")]
        if not conj_steps:
            return None
        a = rng.choice(conj_steps)
        try:
            thm = S.conjunct2(a.thm)
            return ProofStep(next_id, "CONJUNCT2", ("none",), [a.id], thm)
        except S.RuleError:
            return None

    if rule_name == "EQ_MP":
        # Need a |- (a=b) and a |- a.
        eq_steps = [s for s in steps
                    if (s.thm.concl[0] == "Comb"
                         and s.thm.concl[1][0] == "Comb"
                         and s.thm.concl[1][1][0] == "Const"
                         and s.thm.concl[1][1][1] == "=")]
        if not eq_steps:
            return None
        for _ in range(5):
            a = rng.choice(eq_steps)
            lhs = a.thm.concl[1][2]
            matches = [s for s in steps if S.alpha_eq(s.thm.concl, lhs)]
            if not matches:
                continue
            b = rng.choice(matches)
            try:
                thm = S.eq_mp(a.thm, b.thm)
                return ProofStep(next_id, "EQ_MP", ("none",),
                                 [a.id, b.id], thm)
            except S.RuleError:
                continue
        return None

    if rule_name == "INST":
        if not steps:
            return None
        for _ in range(5):
            a = rng.choice(steps)
            fvs = S.free_vars(a.thm.concl)
            if not fvs:
                continue
            # Pick one free var to substitute.
            name, ty = rng.choice(fvs)
            v = T.mk_var(name, ty)
            # Replacement of same type — simplest: a different var.
            pool = (VAR_POOL_NAT if T.type_equal(ty, NAT_TY)
                    else (VAR_POOL_BOOL if T.type_equal(ty, BOOL_TY) else ["x"]))
            rhs_name = rng.choice([p for p in pool if p != name])
            rhs = T.mk_var(rhs_name, ty)
            try:
                thm = S.inst([(v, rhs)], a.thm)
                return ProofStep(next_id, "INST", ("inst", [(v, rhs)]),
                                 [a.id], thm)
            except S.RuleError:
                continue
        return None

    return None  # Unknown rule


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------
RULE_BIAS = {
    # Higher bias = more frequently sampled.  These are tuned to keep
    # the proof depth growing.
    "ASSUME":    3.0,
    "REFL":      2.0,
    "DISCH":     3.0,
    "MP":        2.0,
    "GEN":       1.0,
    "SPEC":      1.0,
    "CONJ":      2.0,
    "CONJUNCT1": 1.0,
    "CONJUNCT2": 1.0,
    "TRANS":     1.5,
    "MK_COMB":   1.0,
    "ABS":       0.8,
    "BETA":      0.5,
    "EQ_MP":     0.8,
    "INST":      0.5,
}


# ---------------------------------------------------------------------------
# Adaptive sampler: inverse-frequency rule choice driven by buffer state.
#
# The fixed RULE_BIAS sampler collapses onto ASSUME/REFL at position 0
# because their applicability is permissive and their prior is high.
# `GeneratorState` carries:
#   - `freq`:      per-position buffer counts (see synth/freq_counter.py)
#   - `attempts`/`successes`: this-round counters used to damp wasted
#     samples on rules that are applicable but structurally hard.
#
# Sampling weight per applicable rule r at position i:
#     score(r) = RULE_BIAS[r] * (succ_rate(r) + ε) / (freq[i][r] + ε)
#   with succ_rate(r) = (successes[r]+1) / (attempts[r]+2)  (Laplace)
# then softmax over log-scores with `temperature`.
# ---------------------------------------------------------------------------
@dataclass
class GeneratorState:
    freq: PositionRuleFreq
    attempts: Dict[str, int] = field(default_factory=dict)
    successes: Dict[str, int] = field(default_factory=dict)
    temperature: float = 1.0


def _has_eq(steps: List[ProofStep]) -> int:
    return sum(1 for s in steps
               if (s.thm.concl and s.thm.concl[0] == "Comb"
                   and s.thm.concl[1][0] == "Comb"
                   and s.thm.concl[1][1][0] == "Const"
                   and s.thm.concl[1][1][1] == "="))


def _has_imp(steps: List[ProofStep]) -> int:
    return sum(1 for s in steps
               if (s.thm.concl and s.thm.concl[0] == "Comb"
                   and s.thm.concl[1][0] == "Comb"
                   and s.thm.concl[1][1][0] == "Const"
                   and s.thm.concl[1][1][1] == "==>"))


def _has_conj(steps: List[ProofStep]) -> int:
    return sum(1 for s in steps
               if (s.thm.concl and s.thm.concl[0] == "Comb"
                   and s.thm.concl[1][0] == "Comb"
                   and s.thm.concl[1][1][0] == "Const"
                   and s.thm.concl[1][1][1] == "/\\"))


def _has_forall(steps: List[ProofStep]) -> int:
    return sum(1 for s in steps
               if (s.thm.concl and s.thm.concl[0] == "Comb"
                   and s.thm.concl[1][0] == "Const"
                   and s.thm.concl[1][1] == "!"))


def _has_bool(steps: List[ProofStep]) -> int:
    return sum(1 for s in steps if S.is_bool(s.thm.concl))


def _applicable(rule: str, steps: List[ProofStep]) -> bool:
    """Cheap necessary-condition check for whether `rule` can possibly
    fire given the current pool of derived theorems.  These are the
    same guards the corresponding `_try_apply` branch performs before
    its `for _ in range(5): try…` retry loop.  Necessary, not
    sufficient — the retry inside `_try_apply` still does the real
    work."""
    if rule in ("REFL", "ASSUME", "BETA"):
        return True
    if rule == "TRANS":      return _has_eq(steps) >= 2
    if rule == "MK_COMB":    return _has_eq(steps) >= 2
    if rule == "ABS":        return _has_eq(steps) >= 1
    if rule == "DISCH":      return len(steps) >= 1
    if rule == "MP":         return _has_imp(steps) >= 1 and len(steps) >= 2
    if rule == "GEN":        return any(S.free_vars(s.thm.concl) for s in steps)
    if rule == "SPEC":       return _has_forall(steps) >= 1
    if rule == "CONJ":       return _has_bool(steps) >= 2
    if rule == "CONJUNCT1":  return _has_conj(steps) >= 1
    if rule == "CONJUNCT2":  return _has_conj(steps) >= 1
    if rule == "EQ_MP":      return _has_eq(steps) >= 1 and len(steps) >= 2
    if rule == "INST":       return any(S.free_vars(s.thm.concl) for s in steps)
    return False


def _sample_rule(state: GeneratorState,
                 position: int,
                 applicable: List[str],
                 rng: random.Random) -> str:
    """Inverse-frequency weighted sample over `applicable`.  Laplace
    smoothing (ε=1.0) in both numerator and denominator avoids
    divide-by-zero on cold rules and keeps single hits from
    dominating.  `RULE_BIAS` is retained as a multiplicative prior so
    hand-tuned knowledge survives."""
    log_scores: List[float] = []
    for r in applicable:
        succ = (state.successes.get(r, 0) + 1.0) / (state.attempts.get(r, 0) + 2.0)
        freq = state.freq.get(position, r) + 1.0
        prior = RULE_BIAS.get(r, 1.0)
        log_scores.append(math.log(prior * succ / freq))
    z = max(log_scores)
    inv_t = 1.0 / max(state.temperature, 1e-6)
    probs = [math.exp((s - z) * inv_t) for s in log_scores]
    return rng.choices(applicable, weights=probs)[0]


def generate_one(rng: random.Random,
                  target_depth: int,
                  max_failures: int = 80,
                  min_random_rules: int = 1,
                  state: Optional[GeneratorState] = None) -> Optional[GenResult]:
    """Generate one random proof, returning the gold (goal, cert) pair.

    target_depth is the TOTAL step count (seeds + random rules).  We
    insist on at least `min_random_rules` non-seed applications, so a
    proof always involves some composition of the seeds, not just the
    bare ASSUME/REFL stack.

    When `state` is supplied, rule selection at each step is driven by
    `_sample_rule` (inverse-frequency adaptive).  The seed-stack
    initialisation still uses ASSUME/REFL since those are the only
    zero-premise rules that produce a non-trivial starting theorem;
    the bias kicks in from the third step onwards.

    Returns None if the random walk got stuck."""
    steps: List[ProofStep] = []
    # Seed pool.  Under the legacy (state=None) path keep the historical
    # ASSUME-3 / REFL-1 weighting.  Under the adaptive path, sample the
    # seed rule from {REFL, ASSUME, BETA} (the only zero-premise rules)
    # via _sample_rule, which routes position-0 frequency feedback
    # through the same inverse-frequency mechanism — without this, the
    # seed-pool hardcoding pins pathology #1 in place.
    seed_count = rng.choice([1, 2])
    seed_zero_premise = ("REFL", "ASSUME", "BETA")
    for _ in range(seed_count):
        if state is not None:
            seed_apps = [r for r in seed_zero_premise if _applicable(r, steps)]
            if not seed_apps:
                break
            kind = _sample_rule(state, len(steps), seed_apps, rng)
            state.attempts[kind] = state.attempts.get(kind, 0) + 1
            s = _try_apply(kind, steps, rng)
            if s is not None:
                state.successes[kind] = state.successes.get(kind, 0) + 1
                state.freq.bump(len(steps), kind)
                steps.append(s)
        else:
            kind = rng.choices(["ASSUME", "REFL"], weights=[3.0, 1.0])[0]
            s = _try_apply(kind, steps, rng)
            if s is not None:
                steps.append(s)
    if not steps:
        return None

    failures = 0
    rules = list(RULE_BIAS.keys())
    weights = [RULE_BIAS[r] for r in rules]

    # Insist on at least min_random_rules additional rule applications.
    rounds_done = 0
    needed = max(target_depth - len(steps), min_random_rules)
    while rounds_done < needed:
        if state is not None:
            applicable = [r for r in rules if _applicable(r, steps)]
            if not applicable:
                break
            rule = _sample_rule(state, len(steps), applicable, rng)
            state.attempts[rule] = state.attempts.get(rule, 0) + 1
        else:
            rule = rng.choices(rules, weights=weights)[0]
        s = _try_apply(rule, steps, rng)
        if s is None:
            failures += 1
            if failures > max_failures:
                break
            continue
        steps.append(s)
        if state is not None:
            state.successes[rule] = state.successes.get(rule, 0) + 1
            # Within-round feedback: the proof we just committed bumps
            # its rule's count at the position it appeared, so the
            # 2000-sample batch self-balances rather than the first
            # sample's choices dominating all 2000.
            state.freq.bump(len(steps) - 1, rule)
        rounds_done += 1

    if len(steps) < 2 or rounds_done < min_random_rules:
        return None

    # Goal = conclusion of the last (or "deepest" by id) step.
    target = steps[-1]
    shape = tuple(s.rule for s in steps)
    return GenResult(goal=target.thm.concl, steps=steps, rule_shape=shape)


# ---------------------------------------------------------------------------
# Convert GenResult to T.Cert (token-encodable)
# ---------------------------------------------------------------------------
def to_cert(gr: GenResult) -> T.Cert:
    """Convert our internal ProofStep list to a T.Cert that can be
    encoded by hol_tokenizer.encode_cert."""
    t_steps = [T.Step(id=s.id, rule=s.rule, witness=s.witness,
                       premises=list(s.premises)) for s in gr.steps]
    return T.Cert(steps=t_steps, concl=gr.goal)


def _collect_vars_in_term(term, out_set):
    if term[0] == "Var":
        out_set.add((term[1], _ty_str(term[2])))
    elif term[0] == "Const":
        pass
    elif term[0] == "Comb":
        _collect_vars_in_term(term[1], out_set)
        _collect_vars_in_term(term[2], out_set)
    elif term[0] == "Abs":
        # bound var also counts (so we can rename consistently)
        out_set.add((term[1], _ty_str(term[2])))
        _collect_vars_in_term(term[3], out_set)


def _ty_str(ty):
    if ty[0] == "Tyvar":
        return f"'{ty[1]}"
    return f"{ty[1]}({','.join(_ty_str(a) for a in ty[2])})"


def _rename_in_term(term, table):
    """Apply name renaming table {(old_name, ty_str): new_name} to all
    variables and bound-variable names in term."""
    if term[0] == "Var":
        key = (term[1], _ty_str(term[2]))
        if key in table:
            return ("Var", table[key], term[2])
        return term
    if term[0] == "Const":
        return term
    if term[0] == "Comb":
        return ("Comb",
                _rename_in_term(term[1], table),
                _rename_in_term(term[2], table))
    if term[0] == "Abs":
        key = (term[1], _ty_str(term[2]))
        new_name = table.get(key, term[1])
        return ("Abs", new_name, term[2], _rename_in_term(term[3], table))
    return term


def _rename_in_witness(w, table):
    if w[0] == "none":
        return w
    if w[0] == "term":
        return ("term", _rename_in_term(w[1], table))
    if w[0] == "var":
        old_name, ty = w[1], w[2]
        new_name = table.get((old_name, _ty_str(ty)), old_name)
        return ("var", new_name, ty)
    if w[0] == "inst":
        return ("inst", [(_rename_in_term(v, table),
                           _rename_in_term(rhs, table))
                          for (v, rhs) in w[1]])
    if w[0] == "insttype":
        return w
    if w[0] == "axiom":
        return w
    if w[0] == "bnw":
        return ("bnw", w[1], w[2], _rename_in_term(w[3], table))
    return w


def variant_of(gr: GenResult, rng: random.Random) -> Optional[GenResult]:
    """Produce a renamed variant of a generated proof.  All free
    variables (and consistently-renamed bound variables) are mapped to
    fresh names drawn from the variable pool of matching type.
    """
    # Collect every (name, ty_str) appearing as Var anywhere in the
    # cert (witnesses + goal).
    collected = set()
    for s in gr.steps:
        if s.witness[0] == "term":
            _collect_vars_in_term(s.witness[1], collected)
        elif s.witness[0] == "var":
            collected.add((s.witness[1], _ty_str(s.witness[2])))
        elif s.witness[0] == "inst":
            for v, rhs in s.witness[1]:
                _collect_vars_in_term(v, collected)
                _collect_vars_in_term(rhs, collected)
        elif s.witness[0] == "bnw":
            _collect_vars_in_term(s.witness[3], collected)
        _collect_vars_in_term(s.thm.concl, collected)
        for h in s.thm.hyps:
            _collect_vars_in_term(h, collected)
    _collect_vars_in_term(gr.goal, collected)
    if not collected:
        return gr

    # Build a renaming.  Critical invariant: the renaming must be
    # injective across (name, ty_str) keys.  Two source variables that
    # collide into the same destination name break rules like INST
    # whose witness requires distinct LHS variables — the kernel
    # rejects the cert on reverify and the sample silently disappears,
    # biasing the kept distribution toward variants whose renaming
    # happens to avoid collisions.
    #
    # If we exhaust the pool of fresh names, the right answer is to
    # bail out (return None) rather than fall back to a used name and
    # corrupt the variant.
    table = {}
    used_new: set = set()
    for (old, ty_str) in collected:
        if ty_str.startswith("nat") or ty_str == "nat()":
            pool = VAR_POOL_NAT
        elif ty_str.startswith("bool") or ty_str == "bool()":
            pool = VAR_POOL_BOOL
        else:
            pool = VAR_POOL_NAT + VAR_POOL_BOOL
        # A name is "available" if no other source variable has already
        # been mapped to it.  We also exclude `old` so the rename
        # actually changes the name (otherwise variants degenerate to
        # the original).
        cands = [p for p in pool if p not in used_new and p != old]
        if not cands:
            return None  # no injective renaming possible — skip variant
        new = rng.choice(cands)
        table[(old, ty_str)] = new
        used_new.add(new)

    new_steps = []
    for s in gr.steps:
        new_w = _rename_in_witness(s.witness, table)
        # The derived theorem also needs renaming (we won't re-derive;
        # the kernel does that on its own).  But we keep the thm field
        # consistent for downstream code that uses it.
        new_concl = _rename_in_term(s.thm.concl, table)
        new_hyps = [_rename_in_term(h, table) for h in s.thm.hyps]
        new_thm = S.Thm(hyps=new_hyps, concl=new_concl)
        new_steps.append(ProofStep(s.id, s.rule, new_w,
                                     list(s.premises), new_thm))
    new_goal = _rename_in_term(gr.goal, table)
    return GenResult(goal=new_goal, steps=new_steps,
                     rule_shape=gr.rule_shape)


def encode_pair(gr: GenResult):
    """Encode (cert, goal) pair with synchronised pool slots so the
    kernel's alpha-equivalence comparison succeeds.

    The verifier infers a separate pool header for each side, mapping
    each slot to a synthetic name like 'v0', 'v1', etc.  If the cert
    and goal allocate the same source var to different slots, their
    synthetic names diverge and alpha_eq returns false even though the
    structural terms are identical.

    Fix: encode the goal *first* in its own ctx (it's a simpler term,
    typically fewer vars), then prefill the cert encoder ctx with the
    goal's pool slots so the cert's slot assignments are a superset.

    Returns (cert_toks, goal_toks, c_hdr, g_hdr).
    """
    # Encode the goal alone.
    g_ctx = T._EncCtx()
    T._encode_term(g_ctx, gr.goal)
    goal_toks = list(g_ctx.out)
    g_hdr = T.PoolHeader(g_ctx.tycons[:], g_ctx.tyvars[:],
                         g_ctx.names[:], g_ctx.vars[:])

    # Prefill the cert encoder ctx with goal's pool entries.
    cert = to_cert(gr)
    c_ctx = T._EncCtx()
    c_ctx.tycons = list(g_hdr.tycons)
    c_ctx.tyvars = list(g_hdr.tyvars)
    c_ctx.names  = list(g_hdr.names)
    c_ctx.vars   = list(g_hdr.vars)
    c_ctx.emit(T.LPAREN); c_ctx.emit(T.KW_CERT)
    for s in cert.steps:
        T._encode_step(c_ctx, s)
    c_ctx.emit(T.LPAREN); c_ctx.emit(T.KW_CONCL)
    c_ctx.emit(T.QUOTE); T._encode_term(c_ctx, cert.concl); c_ctx.emit(T.QUOTE)
    c_ctx.emit(T.RPAREN); c_ctx.emit(T.RPAREN)
    cert_toks = list(c_ctx.out)
    c_hdr = T.PoolHeader(c_ctx.tycons[:], c_ctx.tyvars[:],
                         c_ctx.names[:], c_ctx.vars[:])
    return cert_toks, goal_toks, c_hdr, g_hdr

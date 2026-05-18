"""synth/shadow.py — Python shadow of the kernel rules for fast
synthetic-corpus generation.

The OCaml kernel is the source of truth.  This shadow exists ONLY so
we can generate 10k–100k random proofs without paying subprocess
latency per rule application.  Every emitted proof is re-verified
through the OCaml kernel before it enters the training corpus —
if the shadow ever produces a non-kernel-valid proof, the
re-verification catches it and the sample is discarded.

The shadow tracks just enough term semantics to apply the rules and
detect side-condition failures:

  - alpha-equivalence (already in hol_tokenizer)
  - capture-avoiding substitution of free variables
  - free-variable extraction
  - type substitution

Theorems are (hypothesis-set, conclusion) pairs where the hypothesis
set is deduplicated up to alpha-equivalence.

Rules supported: REFL, TRANS, MK_COMB, ABS, BETA, ASSUME, EQ_MP,
DEDUCT_ANTISYM_RULE, INST, INST_TYPE, GEN, SPEC, CONJ, CONJUNCT1,
CONJUNCT2, MP, DISCH.  (The 17 most-used; EXISTS/CHOOSE/AXIOM/
ETA_AX/SELECT_AX/EM_AX are out of scope for the v1 generator.)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tokenizer", "python"))

import hol_tokenizer as T


# ---------------------------------------------------------------------------
# Term machinery
# ---------------------------------------------------------------------------
def free_vars(t) -> List[Tuple[str, tuple]]:
    """Return ordered list of (name, type) for free variables in t."""
    seen: Set[Tuple[str, tuple]] = set()
    out: List[Tuple[str, tuple]] = []

    def go(t, bound):
        if t[0] == "Var":
            name, ty = t[1], t[2]
            if (name, _ty_key(ty)) not in bound and (name, _ty_key(ty)) not in seen:
                seen.add((name, _ty_key(ty)))
                out.append((name, ty))
        elif t[0] == "Const":
            pass
        elif t[0] == "Comb":
            go(t[1], bound); go(t[2], bound)
        elif t[0] == "Abs":
            go(t[3], bound | {(t[1], _ty_key(t[2]))})
    go(t, set())
    return out


def _ty_key(ty) -> tuple:
    """Hashable key for a type."""
    if ty[0] == "Tyvar":
        return ("Tyvar", ty[1])
    return ("Tyapp", ty[1], tuple(_ty_key(a) for a in ty[2]))


def has_free(name: str, ty: tuple, term) -> bool:
    """Is (name : ty) free in term?"""
    for n, t in free_vars(term):
        if n == name and T.type_equal(t, ty):
            return True
    return False


def gen_fresh_var(name_base: str, ty, avoid_terms: List[tuple]) -> str:
    """Generate a fresh variable name not free in any of avoid_terms."""
    forbidden = set()
    for at in avoid_terms:
        for n, _ in free_vars(at):
            forbidden.add(n)
    if name_base not in forbidden:
        return name_base
    i = 0
    while True:
        cand = f"{name_base}{i}"
        if cand not in forbidden:
            return cand
        i += 1


def subst(theta: List[Tuple[tuple, tuple]], term) -> tuple:
    """Capture-avoiding free-variable substitution.  theta is a list of
    (Var-term, replacement-term) pairs; each Var must have type matching
    its replacement (the kernel checks this).
    """
    # Quick path for empty subst.
    if not theta:
        return term
    # Pre-extract (name, ty_key) → replacement.
    table = {(v[1], _ty_key(v[2])): r for (v, r) in theta if v[0] == "Var"}

    def go(t, bound):
        if t[0] == "Var":
            key = (t[1], _ty_key(t[2]))
            if key in table and key not in bound:
                return table[key]
            return t
        if t[0] == "Const":
            return t
        if t[0] == "Comb":
            return ("Comb", go(t[1], bound), go(t[2], bound))
        if t[0] == "Abs":
            name, ty, body = t[1], t[2], t[3]
            new_bound = bound | {(name, _ty_key(ty))}
            # Capture check: would a replacement for some variable
            # appearing in body bring (name, ty) into a free position?
            need_rename = False
            for (vname, vty_key), rhs in table.items():
                if (vname, vty_key) in new_bound:
                    continue
                # Is vname free in body, and is `name` free in rhs?
                if has_free(vname, _ty_key_to_ty(vty_key), body) and \
                   has_free(name, ty, rhs):
                    need_rename = True
                    break
            if need_rename:
                # Rename the bound variable to a fresh name.
                fresh = gen_fresh_var(name, ty, [body] + [r for r in table.values()])
                new_var = ("Var", fresh, ty)
                renamed_body = subst([(("Var", name, ty), new_var)], body)
                return ("Abs", fresh, ty, go(renamed_body,
                                             bound | {(fresh, _ty_key(ty))}))
            return ("Abs", name, ty, go(body, new_bound))
        raise ValueError(f"subst: bad term {t}")
    return go(term, set())


def _ty_key_to_ty(key):
    """Reverse of _ty_key — recover a type from its hashable key."""
    if key[0] == "Tyvar":
        return ("Tyvar", key[1])
    return ("Tyapp", key[1], [_ty_key_to_ty(a) for a in key[2]])


def type_subst(theta_ty: List[Tuple[str, tuple]], term) -> tuple:
    """Substitute type variables.  theta_ty: list of (tyvar-name,
    replacement-type)."""
    if not theta_ty:
        return term
    table = dict(theta_ty)

    def go_ty(ty):
        if ty[0] == "Tyvar":
            return table.get(ty[1], ty)
        return ("Tyapp", ty[1], [go_ty(a) for a in ty[2]])

    def go(t):
        if t[0] == "Var":
            return ("Var", t[1], go_ty(t[2]))
        if t[0] == "Const":
            return ("Const", t[1], go_ty(t[2]))
        if t[0] == "Comb":
            return ("Comb", go(t[1]), go(t[2]))
        if t[0] == "Abs":
            return ("Abs", t[1], go_ty(t[2]), go(t[3]))
        raise ValueError(t)
    return go(term)


def type_of(t):
    return T.type_of(t)


def alpha_eq(s, t):
    return T.alpha_eq(s, t)


# ---------------------------------------------------------------------------
# Theorem class
# ---------------------------------------------------------------------------
@dataclass
class Thm:
    hyps: List[tuple] = field(default_factory=list)
    concl: tuple = ()

    def __post_init__(self):
        # Dedup hyps modulo alpha.
        hs: List[tuple] = []
        for h in self.hyps:
            if not any(alpha_eq(h, x) for x in hs):
                hs.append(h)
        self.hyps = hs


def union_hyps(*lists):
    out: List[tuple] = []
    for lst in lists:
        for h in lst:
            if not any(alpha_eq(h, x) for x in out):
                out.append(h)
    return out


def remove_hyps(xs, ys):
    return [h for h in xs if not any(alpha_eq(h, y) for y in ys)]


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
class RuleError(Exception):
    pass


def is_bool(t) -> bool:
    return T.type_equal(type_of(t), T.bool_ty())


# Decompose helpers for connectives.
def dest_eq(t):
    if (t[0] == "Comb" and t[1][0] == "Comb"
            and t[1][1][0] == "Const" and t[1][1][1] == "="):
        return (t[1][2], t[2])
    raise RuleError(f"dest_eq: not an equality {t}")


def dest_imp(t):
    if (t[0] == "Comb" and t[1][0] == "Comb"
            and t[1][1][0] == "Const" and t[1][1][1] == "==>"):
        return (t[1][2], t[2])
    raise RuleError("dest_imp: not implication")


def dest_conj(t):
    if (t[0] == "Comb" and t[1][0] == "Comb"
            and t[1][1][0] == "Const" and t[1][1][1] == "/\\"):
        return (t[1][2], t[2])
    raise RuleError("dest_conj: not conjunction")


def dest_forall(t):
    if (t[0] == "Comb" and t[1][0] == "Const" and t[1][1] == "!"
            and t[2][0] == "Abs"):
        # Returns ((bound_name, bound_ty), body)
        abs_ = t[2]
        return ((abs_[1], abs_[2]), abs_[3])
    raise RuleError("dest_forall: not universal")


def refl(t) -> Thm:
    return Thm(hyps=[], concl=T.mk_eq(t, t))


def trans(th1: Thm, th2: Thm) -> Thm:
    a1, b1 = dest_eq(th1.concl)
    a2, b2 = dest_eq(th2.concl)
    if not alpha_eq(b1, a2):
        raise RuleError("trans: middle terms don't match")
    return Thm(hyps=union_hyps(th1.hyps, th2.hyps),
               concl=T.mk_eq(a1, b2))


def mk_comb(th1: Thm, th2: Thm) -> Thm:
    f1, g1 = dest_eq(th1.concl)
    x1, y1 = dest_eq(th2.concl)
    # Type-check: f1 must be a function whose domain matches x1's type.
    ft = type_of(f1)
    if ft[0] != "Tyapp" or ft[1] != "fun":
        raise RuleError("mk_comb: lhs not function")
    if not T.type_equal(ft[2][0], type_of(x1)):
        raise RuleError("mk_comb: type mismatch")
    return Thm(hyps=union_hyps(th1.hyps, th2.hyps),
               concl=T.mk_eq(T.mk_comb(f1, x1), T.mk_comb(g1, y1)))


def abs_rule(v_var, th: Thm) -> Thm:
    if v_var[0] != "Var":
        raise RuleError("abs: not a variable")
    name, ty = v_var[1], v_var[2]
    for h in th.hyps:
        if has_free(name, ty, h):
            raise RuleError("abs: var free in hypotheses")
    s, t = dest_eq(th.concl)
    return Thm(hyps=th.hyps,
               concl=T.mk_eq(("Abs", name, ty, s), ("Abs", name, ty, t)))


def beta(redex) -> Thm:
    if redex[0] != "Comb" or redex[1][0] != "Abs":
        raise RuleError("beta: not (λv. t) x")
    f = redex[1]; arg = redex[2]
    name, ty, body = f[1], f[2], f[3]
    if not (arg[0] == "Var" and arg[1] == name and T.type_equal(arg[2], ty)):
        raise RuleError("beta: argument must equal bound variable")
    return Thm(hyps=[], concl=T.mk_eq(redex, body))


def assume(p) -> Thm:
    if not is_bool(p):
        raise RuleError("assume: not bool")
    return Thm(hyps=[p], concl=p)


def eq_mp(th1: Thm, th2: Thm) -> Thm:
    p1, q1 = dest_eq(th1.concl)
    if not alpha_eq(p1, th2.concl):
        raise RuleError("eq_mp: th2 doesn't match LHS of th1")
    return Thm(hyps=union_hyps(th1.hyps, th2.hyps), concl=q1)


def deduct_antisym(th1: Thm, th2: Thm) -> Thm:
    p, q = th1.concl, th2.concl
    h1 = remove_hyps(th1.hyps, [q])
    h2 = remove_hyps(th2.hyps, [p])
    return Thm(hyps=union_hyps(h1, h2), concl=T.mk_eq(p, q))


def inst(theta, th: Thm) -> Thm:
    return Thm(hyps=[subst(theta, h) for h in th.hyps],
               concl=subst(theta, th.concl))


def inst_type(theta_ty, th: Thm) -> Thm:
    return Thm(hyps=[type_subst(theta_ty, h) for h in th.hyps],
               concl=type_subst(theta_ty, th.concl))


def gen(v_var, th: Thm) -> Thm:
    if v_var[0] != "Var":
        raise RuleError("gen: not a variable")
    name, ty = v_var[1], v_var[2]
    for h in th.hyps:
        if has_free(name, ty, h):
            raise RuleError("gen: var free in hyps")
    return Thm(hyps=th.hyps,
               concl=T.mk_forall(name, ty, th.concl))


def spec(t, th: Thm) -> Thm:
    (n, ty), body = dest_forall(th.concl)
    if not T.type_equal(type_of(t), ty):
        raise RuleError("spec: type mismatch")
    return Thm(hyps=th.hyps,
               concl=subst([(("Var", n, ty), t)], body))


def conj(th1: Thm, th2: Thm) -> Thm:
    return Thm(hyps=union_hyps(th1.hyps, th2.hyps),
               concl=T.mk_conj(th1.concl, th2.concl))


def conjunct1(th: Thm) -> Thm:
    a, _b = dest_conj(th.concl)
    return Thm(hyps=th.hyps, concl=a)


def conjunct2(th: Thm) -> Thm:
    _a, b = dest_conj(th.concl)
    return Thm(hyps=th.hyps, concl=b)


def mp(th1: Thm, th2: Thm) -> Thm:
    p, q = dest_imp(th1.concl)
    if not alpha_eq(p, th2.concl):
        raise RuleError("mp: antecedent mismatch")
    return Thm(hyps=union_hyps(th1.hyps, th2.hyps), concl=q)


def disch(p, th: Thm) -> Thm:
    if not is_bool(p):
        raise RuleError("disch: not bool")
    return Thm(hyps=remove_hyps(th.hyps, [p]),
               concl=T.mk_imp(p, th.concl))


# Rule registry: name -> (apply_fn, arity_premises, witness_kind)
# witness_kind:
#   "term":  takes a term witness
#   "var":   takes a (var, ty)
#   "inst":  takes a list of (var, term) pairs
#   "insttype": takes a list of (tyvar_name, ty)
#   "none":  no witness
RULES = {
    "REFL":       (lambda w, ps: refl(w),                             0, "term"),
    "TRANS":      (lambda w, ps: trans(ps[0], ps[1]),                 2, "none"),
    "MK_COMB":    (lambda w, ps: mk_comb(ps[0], ps[1]),               2, "none"),
    "ABS":        (lambda w, ps: abs_rule(w, ps[0]),                  1, "var"),
    "BETA":       (lambda w, ps: beta(w),                             0, "term"),
    "ASSUME":     (lambda w, ps: assume(w),                           0, "term"),
    "EQ_MP":      (lambda w, ps: eq_mp(ps[0], ps[1]),                 2, "none"),
    "DEDUCT_ANTISYM_RULE": (lambda w, ps: deduct_antisym(ps[0], ps[1]), 2, "none"),
    "INST":       (lambda w, ps: inst(w, ps[0]),                      1, "inst"),
    "INST_TYPE":  (lambda w, ps: inst_type(w, ps[0]),                 1, "insttype"),
    "GEN":        (lambda w, ps: gen(w, ps[0]),                       1, "var"),
    "SPEC":       (lambda w, ps: spec(w, ps[0]),                      1, "term"),
    "CONJ":       (lambda w, ps: conj(ps[0], ps[1]),                  2, "none"),
    "CONJUNCT1":  (lambda w, ps: conjunct1(ps[0]),                    1, "none"),
    "CONJUNCT2":  (lambda w, ps: conjunct2(ps[0]),                    1, "none"),
    "MP":         (lambda w, ps: mp(ps[0], ps[1]),                    2, "none"),
    "DISCH":      (lambda w, ps: disch(w, ps[0]),                     1, "term"),
}

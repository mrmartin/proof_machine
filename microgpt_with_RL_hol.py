"""
microgpt_with_RL_hol.py — train a small GPT to find HOL proofs via REINFORCE.

Adapted from microgpt_with_RL.py (Karpathy's microgpt + RL conversion).
The structural changes:
  - vocab: replaced with the HOL certificate vocab (lexeme-level, 232 ids).
  - seed prompts: HOL theorems (encoded as token sequences).
  - validator V: grammar-aware certificate checker, with concl-alpha-match.
  - sampling: masked by the grammar automaton so the policy can only emit
    syntactically valid certificate prefixes.
  - **Data-parallel rollouts**: N workers (default `cpu_count()`) each
    sample one rollout per RL step; main averages gradients and applies
    one Adam update.

Adam and the model code are otherwise untouched.

Reward schedule (terminal) — configurable via env vars so we can ablate:
   -1                                grammar didn't accept the completion
    HOL_REWARD_WRONG_CONCL  (def 0)  valid cert, but concl ≠ prompted goal
   +100                              valid cert AND concl alpha-equiv to goal
   + HOL_STEP_BONUS         (def 0)  extra if cert has at least one (step …)

Optional supervised warmup: HOL_WARMUP_STEPS pre-RL passes of teacher-forced
next-token CE over (prompt + gold cert) for each seed.

Files:
  HOL_LOG_PATH   (default hol_rl_run.log)
  HOL_CKPT_PATH  (default hol_rl_ckpt.pkl)
"""

import os
import math
import random
import sys
import time
import pickle
import subprocess
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tokenizer", "python"))
import hol_tokenizer as T

# ---------------------------------------------------------------------------
# Configuration via env vars.
# ---------------------------------------------------------------------------
LOG_PATH       = os.environ.get("HOL_LOG_PATH",  os.path.join(os.path.dirname(__file__), "hol_rl_run.log"))
CKPT_PATH      = os.environ.get("HOL_CKPT_PATH", os.path.join(os.path.dirname(__file__), "hol_rl_ckpt.pkl"))
NUM_STEPS      = int(os.environ.get("HOL_NUM_STEPS",  "8000"))
LOG_EVERY      = int(os.environ.get("HOL_LOG_EVERY",  "20"))
CKPT_EVERY     = int(os.environ.get("HOL_CKPT_EVERY", "200"))
N_WORKERS      = int(os.environ.get("HOL_NUM_WORKERS", "0")) or os.cpu_count()
WARMUP_STEPS   = int(os.environ.get("HOL_WARMUP_STEPS", "0"))
REWARD_WRONG_CONCL = float(os.environ.get("HOL_REWARD_WRONG_CONCL", "0"))
STEP_BONUS         = float(os.environ.get("HOL_STEP_BONUS", "0"))
ENTROPY_BETA       = float(os.environ.get("HOL_ENTROPY_BETA", "0.0"))
VERIFIER_BIN       = os.environ.get(
    "HOL_VERIFIER_BIN",
    os.path.join(os.path.dirname(__file__), "_build", "default", "bin", "verify_tokens.exe"),
)
USE_KERNEL_VERIFY  = int(os.environ.get("HOL_USE_KERNEL_VERIFY", "1"))

log_f = open(LOG_PATH, "a", buffering=1)
def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    log_f.write(line + "\n")

# ---------------------------------------------------------------------------
# Model hyperparams (architecture untouched).
# ---------------------------------------------------------------------------
n_layer = 1
n_embd = 12
n_head = 4
head_dim = n_embd // n_head
block_size = int(os.environ.get("HOL_BLOCK_SIZE", "192"))
max_gen_tokens = int(os.environ.get("HOL_MAX_GEN", "128"))
vocab_size = T.VOCAB_SIZE
BOS = T.BOS
EOS = T.EOS

# ---------------------------------------------------------------------------
# Autograd Value (identical to microgpt.py).
# ---------------------------------------------------------------------------
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    def __init__(self, data, children=(), local_grads=()):
        self.data = data; self.grad = 0
        self._children = children; self._local_grads = local_grads
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    def backward(self):
        topo = []; visited = set()
        def build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._children: build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# ---------------------------------------------------------------------------
# Seeds.
# ---------------------------------------------------------------------------
NAT  = T.nat_ty()
BOOL = T.bool_ty()

def G_refl(name):     v = T.mk_var(name, NAT);  return T.mk_eq(v, v)
def G_refl_bool(name): v = T.mk_var(name, BOOL); return T.mk_eq(v, v)
def G_imp_self(name):  p = T.mk_var(name, BOOL); return T.mk_imp(p, p)

# Five new test seeds.  Domain-flavoured names (prime, gcd, ...) — these
# get canonicalised to slot tokens at encode time, so the surface naming
# doesn't change what the model sees but documents the intended use.
def G_prime_imp(name_pred, name_n):
    """(prime n) ==> (prime n) — uses a unary predicate of nat→bool."""
    pred = T.mk_var(name_pred, T.fun_ty(NAT, BOOL))
    n    = T.mk_var(name_n, NAT)
    p_n  = T.mk_comb(pred, n)
    return T.mk_imp(p_n, p_n)

def G_conj_proj_left(name_p, name_q):
    p = T.mk_var(name_p, BOOL); q = T.mk_var(name_q, BOOL)
    return T.mk_imp(T.mk_conj(p, q), p)

def G_conj_swap(name_p, name_q):
    p = T.mk_var(name_p, BOOL); q = T.mk_var(name_q, BOOL)
    return T.mk_imp(T.mk_conj(p, q), T.mk_conj(q, p))

def G_k_combinator(name_p, name_q):
    p = T.mk_var(name_p, BOOL); q = T.mk_var(name_q, BOOL)
    return T.mk_imp(p, T.mk_imp(q, p))

def G_gcd_refl(name_g, name_a, name_b):
    """gcd a b = gcd a b — binary function on nat tests curried application."""
    gty = T.fun_ty(NAT, T.fun_ty(NAT, NAT))
    g = T.mk_var(name_g, gty)
    a = T.mk_var(name_a, NAT); b = T.mk_var(name_b, NAT)
    g_a_b = T.mk_comb(T.mk_comb(g, a), b)
    return T.mk_eq(g_a_b, g_a_b)

SEEDS = [
    # Five original logical-tautology seeds.
    ("refl_x",          G_refl("x"),                       1),
    ("refl_y",          G_refl("y"),                       1),
    ("refl_n",          G_refl("n"),                       1),
    ("refl_p_bool",     G_refl_bool("p"),                  1),
    ("refl_q_bool",     G_refl_bool("q"),                  1),
    ("imp_p",           G_imp_self("p"),                   2),
    ("imp_q",           G_imp_self("q"),                   2),
    # Five new seeds with number-theory / algebra / crypto flavour.
    ("prime_imp_prime", G_prime_imp("prime", "n"),         2),
    ("conj_proj_left",  G_conj_proj_left("a", "b"),        3),
    ("conj_swap",       G_conj_swap("a", "b"),             5),
    ("k_combinator",    G_k_combinator("p", "q"),          3),
    ("gcd_refl",        G_gcd_refl("gcd", "a", "b"),       1),
]

def gold_cert_for(label, goal):
    """Hand-built kernel-valid cert that proves `goal`.  Used for the
    supervised warmup (tweak #3)."""
    if label.startswith("refl_") or label == "gcd_refl":
        # goal = mk_eq(x, x) = ("Comb", ("Comb", "=", x), x)
        x = goal[2]
        return T.Cert(
            steps=[T.Step(1, "REFL", ("term", x), [])],
            concl=goal,
        )
    if label.startswith("imp_") or label == "prime_imp_prime":
        # goal = mk_imp(p, p) where p may itself be compound (e.g. a predicate
        # application).  Extract p from the consequent slot of the outer Comb.
        p = goal[2]
        return T.Cert(
            steps=[
                T.Step(1, "ASSUME", ("term", p), []),
                T.Step(2, "DISCH",  ("term", p), [1]),
            ],
            concl=goal,
        )
    if label == "conj_proj_left":
        # goal = mk_imp(p /\ q, p)
        pq = goal[1][2]          # the antecedent (p /\ q)
        return T.Cert(
            steps=[
                T.Step(1, "ASSUME",    ("term", pq), []),
                T.Step(2, "CONJUNCT1", ("none",),    [1]),
                T.Step(3, "DISCH",     ("term", pq), [2]),
            ],
            concl=goal,
        )
    if label == "conj_swap":
        # goal = mk_imp(p /\ q, q /\ p)
        pq = goal[1][2]
        return T.Cert(
            steps=[
                T.Step(1, "ASSUME",    ("term", pq), []),
                T.Step(2, "CONJUNCT1", ("none",),    [1]),
                T.Step(3, "CONJUNCT2", ("none",),    [1]),
                T.Step(4, "CONJ",      ("none",),    [3, 2]),
                T.Step(5, "DISCH",     ("term", pq), [4]),
            ],
            concl=goal,
        )
    if label == "k_combinator":
        # goal = mk_imp(p, mk_imp(q, p))
        p = goal[1][2]            # p
        q = goal[2][1][2]         # q (from the inner mk_imp)
        return T.Cert(
            steps=[
                T.Step(1, "ASSUME", ("term", p), []),
                T.Step(2, "DISCH",  ("term", q), [1]),
                T.Step(3, "DISCH",  ("term", p), [2]),
            ],
            concl=goal,
        )
    raise ValueError(f"no gold cert for {label}")

# ---------------------------------------------------------------------------
# Supervised corpus: ~1000 (goal, gold cert) pairs across distinct
# token-patterns.  Each pattern is replicated many times — token-identical
# instances still give SGD more updates against the same loss surface.
# ---------------------------------------------------------------------------

def _build_supervised_corpus():
    """Return a list of (goal_term, gold_cert) pairs.  Each pattern is
    instantiated multiple times to bulk up the warmup data; the model's
    canonical-slot tokenizer means name-variation collapses to the same
    token sequence, so what matters is the distinct token-PATTERN count
    and how many times each gets hit by SGD."""
    nat   = NAT
    bool_ = BOOL
    ind   = T.ind_ty()

    out = []

    def refl_pair(name, ty):
        v = T.mk_var(name, ty)
        return (T.mk_eq(v, v),
                T.Cert(steps=[T.Step(1, "REFL", ("term", v), [])],
                       concl=T.mk_eq(v, v)))

    def imp_pair(name):
        p = T.mk_var(name, bool_)
        return (T.mk_imp(p, p),
                T.Cert(steps=[
                    T.Step(1, "ASSUME", ("term", p), []),
                    T.Step(2, "DISCH",  ("term", p), [1]),
                ], concl=T.mk_imp(p, p)))

    def forall_refl_pair(name, ty):
        v = T.mk_var(name, ty)
        eq = T.mk_eq(v, v)
        return (T.mk_forall(name, ty, eq),
                T.Cert(steps=[
                    T.Step(1, "REFL", ("term", v), []),
                    T.Step(2, "GEN",  ("var", name, ty), [1]),
                ], concl=T.mk_forall(name, ty, eq)))

    def beta_pair(bound_name, _ignored=None):
        # Kernel BETA fires on (λv. body) v where the argument equals the
        # bound variable.  Force witness == bound; legacy callers passed a
        # second name but those certs were kernel-invalid.
        ty = nat
        v = T.mk_var(bound_name, ty)
        lam  = T.mk_abs(bound_name, ty, v)
        app  = T.mk_comb(lam, v)
        return (T.mk_eq(app, v),
                T.Cert(steps=[T.Step(1, "BETA", ("term", app), [])],
                       concl=T.mk_eq(app, v)))

    # --- New patterns (20 total) ------------------------------------------
    #
    # Logical schemata with domain-flavoured naming.  The tokenizer
    # canonicalises variable names to pool slots, so the proof STRUCTURE is
    # what the model learns; the surface vocabulary is for readability.
    # Each function returns a (goal, gold_cert) pair.

    def k_pair(name_p, name_q):
        """p ==> (q ==> p) — K combinator.  3 steps."""
        p = T.mk_var(name_p, bool_); q = T.mk_var(name_q, bool_)
        goal = T.mk_imp(p, T.mk_imp(q, p))
        return (goal, T.Cert(steps=[
            T.Step(1, "ASSUME", ("term", p), []),
            T.Step(2, "DISCH",  ("term", q), [1]),
            T.Step(3, "DISCH",  ("term", p), [2]),
        ], concl=goal))

    def conj_elim1_pair(name_p, name_q):
        """(p /\\ q) ==> p — conjunction elimination (left).  3 steps."""
        p = T.mk_var(name_p, bool_); q = T.mk_var(name_q, bool_)
        pq = T.mk_conj(p, q); goal = T.mk_imp(pq, p)
        return (goal, T.Cert(steps=[
            T.Step(1, "ASSUME",    ("term", pq), []),
            T.Step(2, "CONJUNCT1", ("none",),    [1]),
            T.Step(3, "DISCH",     ("term", pq), [2]),
        ], concl=goal))

    def conj_elim2_pair(name_p, name_q):
        """(p /\\ q) ==> q — conjunction elimination (right).  3 steps."""
        p = T.mk_var(name_p, bool_); q = T.mk_var(name_q, bool_)
        pq = T.mk_conj(p, q); goal = T.mk_imp(pq, q)
        return (goal, T.Cert(steps=[
            T.Step(1, "ASSUME",    ("term", pq), []),
            T.Step(2, "CONJUNCT2", ("none",),    [1]),
            T.Step(3, "DISCH",     ("term", pq), [2]),
        ], concl=goal))

    def conj_commute_pair(name_p, name_q):
        """(p /\\ q) ==> (q /\\ p) — conjunction commutes.  5 steps."""
        p = T.mk_var(name_p, bool_); q = T.mk_var(name_q, bool_)
        pq = T.mk_conj(p, q); qp = T.mk_conj(q, p); goal = T.mk_imp(pq, qp)
        return (goal, T.Cert(steps=[
            T.Step(1, "ASSUME",    ("term", pq), []),
            T.Step(2, "CONJUNCT1", ("none",),    [1]),
            T.Step(3, "CONJUNCT2", ("none",),    [1]),
            T.Step(4, "CONJ",      ("none",),    [3, 2]),
            T.Step(5, "DISCH",     ("term", pq), [4]),
        ], concl=goal))

    def curried_conj_pair(name_p, name_q):
        """p ==> q ==> (p /\\ q) — curried conjunction introduction.  5 steps."""
        p = T.mk_var(name_p, bool_); q = T.mk_var(name_q, bool_)
        pq = T.mk_conj(p, q); goal = T.mk_imp(p, T.mk_imp(q, pq))
        return (goal, T.Cert(steps=[
            T.Step(1, "ASSUME", ("term", p), []),
            T.Step(2, "ASSUME", ("term", q), []),
            T.Step(3, "CONJ",   ("none",),   [1, 2]),
            T.Step(4, "DISCH",  ("term", q), [3]),
            T.Step(5, "DISCH",  ("term", p), [4]),
        ], concl=goal))

    def self_conj_pair(name_p):
        """p ==> (p /\\ p) — same-prop conjunction.  3 steps."""
        p = T.mk_var(name_p, bool_)
        pp = T.mk_conj(p, p); goal = T.mk_imp(p, pp)
        return (goal, T.Cert(steps=[
            T.Step(1, "ASSUME", ("term", p), []),
            T.Step(2, "CONJ",   ("none",),   [1, 1]),
            T.Step(3, "DISCH",  ("term", p), [2]),
        ], concl=goal))

    def imp_self_conj_pair(name_p, name_q):
        """(p /\\ q) ==> (p /\\ q) — imp_self with compound antecedent.  2 steps."""
        p = T.mk_var(name_p, bool_); q = T.mk_var(name_q, bool_)
        pq = T.mk_conj(p, q); goal = T.mk_imp(pq, pq)
        return (goal, T.Cert(steps=[
            T.Step(1, "ASSUME", ("term", pq), []),
            T.Step(2, "DISCH",  ("term", pq), [1]),
        ], concl=goal))

    def forall_double_pair(name_x, name_y, ty):
        """∀x. ∀y. x = x — nested quantifiers.  3 steps."""
        x = T.mk_var(name_x, ty); eq = T.mk_eq(x, x)
        inner = T.mk_forall(name_y, ty, eq)
        outer = T.mk_forall(name_x, ty, inner)
        return (outer, T.Cert(steps=[
            T.Step(1, "REFL", ("term", x),            []),
            T.Step(2, "GEN",  ("var", name_y, ty),    [1]),
            T.Step(3, "GEN",  ("var", name_x, ty),    [2]),
        ], concl=outer))

    def spec_instance_pair(name_v, name_n, ty):
        """(∀v. v = v) ==> n = n — universal instantiation.  3 steps."""
        v = T.mk_var(name_v, ty); eq_v = T.mk_eq(v, v)
        forall_eq = T.mk_forall(name_v, ty, eq_v)
        n = T.mk_var(name_n, ty); eq_n = T.mk_eq(n, n)
        goal = T.mk_imp(forall_eq, eq_n)
        return (goal, T.Cert(steps=[
            T.Step(1, "ASSUME", ("term", forall_eq), []),
            T.Step(2, "SPEC",   ("term", n),         [1]),
            T.Step(3, "DISCH",  ("term", forall_eq), [2]),
        ], concl=goal))

    def trans_refl_pair(name_a, ty):
        """a = a via TRANS of two REFLs — structural variety.  3 steps."""
        a = T.mk_var(name_a, ty); goal = T.mk_eq(a, a)
        return (goal, T.Cert(steps=[
            T.Step(1, "REFL",  ("term", a), []),
            T.Step(2, "REFL",  ("term", a), []),
            T.Step(3, "TRANS", ("none",),   [1, 2]),
        ], concl=goal))

    def mk_comb_refl_pair(name_f, name_a):
        """f a = f a via MK_COMB(REFL f, REFL a) — function-typed variable.  3 steps."""
        fty = T.fun_ty(nat, nat)
        f = T.mk_var(name_f, fty); a = T.mk_var(name_a, nat)
        fa = T.mk_comb(f, a); goal = T.mk_eq(fa, fa)
        return (goal, T.Cert(steps=[
            T.Step(1, "REFL",    ("term", f), []),
            T.Step(2, "REFL",    ("term", a), []),
            T.Step(3, "MK_COMB", ("none",),   [1, 2]),
        ], concl=goal))

    def abs_refl_pair(name_x, _ignored=None):
        """(λx. x) = (λx. x) — identity-lambda equality via REFL+ABS.  2 steps.
        Binder and body variable must share a name to avoid a pool-slot
        ordering mismatch between cert (REFL-first) and goal (binder-first)
        encodings — they'd otherwise canonicalise the same logical variable
        to different free slots and fail kernel α-eq."""
        v = T.mk_var(name_x, nat)
        lam = T.mk_abs(name_x, nat, v); goal = T.mk_eq(lam, lam)
        return (goal, T.Cert(steps=[
            T.Step(1, "REFL", ("term", v),            []),
            T.Step(2, "ABS",  ("var", name_x, nat),   [1]),
        ], concl=goal))

    def gen_then_imp_pair(name_p, name_x):
        """p ==> ∀x:nat. p — vacuous universal under hypothesis.  3 steps."""
        p = T.mk_var(name_p, bool_)
        forall_p = T.mk_forall(name_x, nat, p)
        goal = T.mk_imp(p, forall_p)
        return (goal, T.Cert(steps=[
            T.Step(1, "ASSUME", ("term", p),         []),
            T.Step(2, "GEN",    ("var", name_x, nat), [1]),
            T.Step(3, "DISCH",  ("term", p),         [2]),
        ], concl=goal))

    def exists_intro_pair(name_y, name_w, ty):
        """∃y. y = y via EXISTS-intro of REFL(w).  2 steps.
        Requires name_y ≠ name_w (EXISTS needs the bound name to not clash
        with a free variable in the conclusion of the premise theorem)."""
        w = T.mk_var(name_w, ty)
        body = T.mk_eq(T.mk_var(name_y, ty), T.mk_var(name_y, ty))
        goal = T.mk_exists(name_y, ty, body)
        return (goal, T.Cert(steps=[
            T.Step(1, "REFL",   ("term", w),                       []),
            T.Step(2, "EXISTS", ("bnw", name_y, ty, w),            [1]),
        ], concl=goal))

    def mp_curried_pair(name_p, name_q):
        """(p ==> q) ==> p ==> q — uses MP.  5 steps."""
        p = T.mk_var(name_p, bool_); q = T.mk_var(name_q, bool_)
        pimpq = T.mk_imp(p, q)
        goal = T.mk_imp(pimpq, T.mk_imp(p, q))
        return (goal, T.Cert(steps=[
            T.Step(1, "ASSUME", ("term", pimpq), []),
            T.Step(2, "ASSUME", ("term", p),     []),
            T.Step(3, "MP",     ("none",),       [1, 2]),
            T.Step(4, "DISCH",  ("term", p),     [3]),
            T.Step(5, "DISCH",  ("term", pimpq), [4]),
        ], concl=goal))

    def deduct_antisym_pair(name_p):
        """p = p via DEDUCT_ANTISYM of ASSUME(p), ASSUME(p).  3 steps."""
        p = T.mk_var(name_p, bool_); goal = T.mk_eq(p, p)
        return (goal, T.Cert(steps=[
            T.Step(1, "ASSUME",              ("term", p), []),
            T.Step(2, "ASSUME",              ("term", p), []),
            T.Step(3, "DEDUCT_ANTISYM_RULE", ("none",),   [1, 2]),
        ], concl=goal))

    def refl_fun_pair(name_f, dom_ty, cod_ty):
        """f = f for f of function type — wider type variety.  1 step."""
        f = T.mk_var(name_f, T.fun_ty(dom_ty, cod_ty))
        goal = T.mk_eq(f, f)
        return (goal, T.Cert(steps=[
            T.Step(1, "REFL", ("term", f), []),
        ], concl=goal))

    def refl_app_pair(name_pred, name_n):
        """(pred n) = (pred n) for pred:nat→bool — predicate-application REFL.  1 step."""
        pred = T.mk_var(name_pred, T.fun_ty(nat, bool_))
        n = T.mk_var(name_n, nat)
        p_n = T.mk_comb(pred, n)
        goal = T.mk_eq(p_n, p_n)
        return (goal, T.Cert(steps=[
            T.Step(1, "REFL", ("term", p_n), []),
        ], concl=goal))

    def imp_self_app_pair(name_pred, name_n):
        """(p n) ==> (p n) — imp_self over a predicate application.  2 steps.
        Bridges the imp_self structure to predicate-application prompts."""
        pred = T.mk_var(name_pred, T.fun_ty(nat, bool_))
        n = T.mk_var(name_n, nat)
        p_n = T.mk_comb(pred, n)
        goal = T.mk_imp(p_n, p_n)
        return (goal, T.Cert(steps=[
            T.Step(1, "ASSUME", ("term", p_n), []),
            T.Step(2, "DISCH",  ("term", p_n), [1]),
        ], concl=goal))

    def refl_curried_pair(name_f, name_a, name_b):
        """(f a b) = (f a b) for f:nat→nat→nat — curried 2-arg application REFL.
        1 step.  Bridges to gcd-style binary-operator prompts."""
        fty = T.fun_ty(nat, T.fun_ty(nat, nat))
        f = T.mk_var(name_f, fty)
        a = T.mk_var(name_a, nat); b = T.mk_var(name_b, nat)
        fab = T.mk_comb(T.mk_comb(f, a), b)
        goal = T.mk_eq(fab, fab)
        return (goal, T.Cert(steps=[
            T.Step(1, "REFL", ("term", fab), []),
        ], concl=goal))

    def imp_eq_pair(name_m, name_n, ty):
        """(m = n) ==> (m = n) — imp_self with equation antecedent.  2 steps."""
        m = T.mk_var(name_m, ty); n = T.mk_var(name_n, ty)
        eq = T.mk_eq(m, n); goal = T.mk_imp(eq, eq)
        return (goal, T.Cert(steps=[
            T.Step(1, "ASSUME", ("term", eq), []),
            T.Step(2, "DISCH",  ("term", eq), [1]),
        ], concl=goal))

    NAMES = ["a", "b", "c", "p", "q", "x", "y", "z", "n", "m", "k", "u", "v", "w"]
    per_pattern = int(os.environ.get("HOL_CORPUS_PER_PATTERN", "200"))

    def _pick(i):  return NAMES[i % len(NAMES)]
    def _pick2(i): return NAMES[(i + 1) % len(NAMES)]

    # --- Original 6 patterns ---------------------------------------------
    for ty in (nat, bool_, ind):
        for _ in range(per_pattern):
            out.append(refl_pair(_pick(len(out)), ty))

    for _ in range(per_pattern * 2):
        out.append(imp_pair(_pick(len(out))))

    for ty in (nat, bool_):
        for _ in range(per_pattern):
            out.append(forall_refl_pair(_pick(len(out)), ty))

    for _ in range(per_pattern):
        out.append(beta_pair("x", _pick(len(out))))

    # --- New patterns (per_pattern_new replicas each) --------------------
    # Use full per_pattern replication so each new pattern gets equal
    # sampling weight to the original ones; the rarer 5-step proof shapes
    # otherwise get half the SGD updates and fall behind under greedy eval.
    per_pattern_new = per_pattern

    for _ in range(per_pattern_new):
        out.append(k_pair(_pick(len(out)), _pick2(len(out))))
    for _ in range(per_pattern_new):
        out.append(conj_elim1_pair(_pick(len(out)), _pick2(len(out))))
    for _ in range(per_pattern_new):
        out.append(conj_elim2_pair(_pick(len(out)), _pick2(len(out))))
    for _ in range(per_pattern_new):
        out.append(conj_commute_pair(_pick(len(out)), _pick2(len(out))))
    for _ in range(per_pattern_new):
        out.append(curried_conj_pair(_pick(len(out)), _pick2(len(out))))
    for _ in range(per_pattern_new):
        out.append(self_conj_pair(_pick(len(out))))
    for _ in range(per_pattern_new):
        out.append(imp_self_conj_pair(_pick(len(out)), _pick2(len(out))))
    for ty in (nat, bool_):
        for _ in range(per_pattern_new):
            out.append(forall_double_pair(_pick(len(out)), _pick2(len(out)), ty))
    for ty in (nat, bool_):
        for _ in range(per_pattern_new):
            out.append(spec_instance_pair(_pick(len(out)), _pick2(len(out)), ty))
    for ty in (nat, bool_, ind):
        for _ in range(per_pattern_new):
            out.append(trans_refl_pair(_pick(len(out)), ty))
    for _ in range(per_pattern_new):
        out.append(mk_comb_refl_pair(_pick(len(out)), _pick2(len(out))))
    for _ in range(per_pattern_new):
        out.append(abs_refl_pair(_pick(len(out)), _pick2(len(out))))
    for _ in range(per_pattern_new):
        out.append(gen_then_imp_pair(_pick(len(out)), _pick2(len(out))))
    for ty in (nat, bool_):
        for _ in range(per_pattern_new):
            out.append(exists_intro_pair(_pick(len(out)), _pick2(len(out)), ty))
    for _ in range(per_pattern_new):
        out.append(mp_curried_pair(_pick(len(out)), _pick2(len(out))))
    for _ in range(per_pattern_new):
        out.append(deduct_antisym_pair(_pick(len(out))))
    for dom, cod in ((nat, nat), (nat, bool_), (bool_, bool_)):
        for _ in range(per_pattern_new):
            out.append(refl_fun_pair(_pick(len(out)), dom, cod))
    for _ in range(per_pattern_new):
        out.append(refl_app_pair(_pick(len(out)), _pick2(len(out))))
    for _ in range(per_pattern_new):
        out.append(imp_self_app_pair(_pick(len(out)), _pick2(len(out))))
    for _ in range(per_pattern_new):
        out.append(refl_curried_pair(_pick(len(out)), _pick2(len(out)),
                                      NAMES[(len(out) + 2) % len(NAMES)]))
    for ty in (nat, bool_):
        for _ in range(per_pattern_new):
            out.append(imp_eq_pair(_pick(len(out)), _pick2(len(out)), ty))

    random.Random(42).shuffle(out)
    return out

CORPUS = _build_supervised_corpus()

# Pre-encode every corpus entry once.  goal_toks for the prompt,
# cert_toks for the supervised target.
ENCODED_CORPUS = []
for g, c in CORPUS:
    gt, _ = T.encode_term_only(g)
    ct, _ = T.encode_cert(c)
    if len(gt) + len(ct) + 3 <= block_size:
        ENCODED_CORPUS.append((gt, ct))

# Build prompt tokens + gold cert tokens for the TEST seeds (used by RL
# phase + final inference, unchanged from before).
encoded_goals = []
for label, goal, _ in SEEDS:
    goal_toks, _ = T.encode_term_only(goal)
    gold = gold_cert_for(label, goal)
    cert_toks, _ = T.encode_cert(gold)
    encoded_goals.append((label, goal, goal_toks, cert_toks))

# ---------------------------------------------------------------------------
# Worker globals — set by _worker_init in each Pool process.
# ---------------------------------------------------------------------------
_W_state_dict = None
_W_params = None
_W_n_params = None
_W_verifier = None       # subprocess.Popen running the OCaml verify_tokens.exe

def _build_model():
    """Build the model structure with placeholder zero weights.  Used in
    workers (and as main's structure)."""
    matrix = lambda nout, nin: [[Value(0.0) for _ in range(nin)] for _ in range(nout)]
    sd = {
        'wte':     matrix(vocab_size, n_embd),
        'wpe':     matrix(block_size, n_embd),
        'lm_head': matrix(vocab_size, n_embd),
    }
    for i in range(n_layer):
        sd[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
        sd[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
    flat = [p for mat in sd.values() for row in mat for p in row]
    return sd, flat

def _gpt(sd, token_id, pos_id, keys, values):
    tok_emb = sd['wte'][token_id]
    pos_emb = sd['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    def rmsnorm(x):
        ms = sum(xi * xi for xi in x) / len(x)
        scale = (ms + 1e-5) ** -0.5
        return [xi * scale for xi in x]
    def linear(x, w):
        return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
    def softmax(logits):
        max_val = max(val.data for val in logits)
        exps = [(val - max_val).exp() for val in logits]
        total = sum(exps)
        return [e / total for e in exps]
    x = rmsnorm(x)
    for li in range(n_layer):
        x_residual = x; x = rmsnorm(x)
        q = linear(x, sd[f'layer{li}.attn_wq'])
        k = linear(x, sd[f'layer{li}.attn_wk'])
        v = linear(x, sd[f'layer{li}.attn_wv'])
        keys[li].append(k); values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                           for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                        for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, sd[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x; x = rmsnorm(x)
        x = linear(x, sd[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, sd[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
    logits = linear(x, sd['lm_head'])
    return logits

def _softmax_data(logits):
    mv = max(v.data for v in logits)
    exps = [(v - mv).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def sample_with_mask(logits, mask):
    """Sample from softmax(masked logits).  Returns (tok_id, log_prob, entropy)
    where log_prob and entropy are Value objects (autograd-aware)."""
    valid_idx = [i for i, ok in enumerate(mask) if ok]
    valid_logits = [logits[i] for i in valid_idx]
    max_v = max(v.data for v in valid_logits)
    exps = [(v - max_v).exp() for v in valid_logits]
    total = sum(exps)
    probs = [e / total for e in exps]
    # entropy = -Σ p log p
    entropy = sum(-(p * p.log()) for p in probs)
    k = random.choices(range(len(probs)), weights=[p.data for p in probs])[0]
    return valid_idx[k], probs[k].log(), entropy

def argmax_with_mask(logits, mask):
    best, best_v = None, -1e18
    for i, ok in enumerate(mask):
        if ok and logits[i].data > best_v:
            best, best_v = i, logits[i].data
    return best

# ---------------------------------------------------------------------------
# Validator V.
# ---------------------------------------------------------------------------
def _infer_header(toks):
    hdr = T.PoolHeader()
    seen_v, seen_n, seen_tc, seen_tv = set(), set(), set(), set()
    for t in toks:
        if T.is_var(t):
            k = t - T.VAR_FIRST
            if k not in seen_v:
                seen_v.add(k)
                while len(hdr.vars) <= k: hdr.vars.append(f"v{len(hdr.vars)}")
        elif T.is_name(t):
            k = t - T.NAME_FIRST
            if k not in seen_n:
                seen_n.add(k)
                while len(hdr.names) <= k: hdr.names.append(f"n{len(hdr.names)}")
        elif T.is_tycon(t):
            k = t - T.TYCON_FIRST
            if k not in seen_tc:
                seen_tc.add(k)
                while len(hdr.tycons) <= k: hdr.tycons.append(f"t{len(hdr.tycons)}")
        elif T.is_tyvar(t):
            k = t - T.TYVAR_FIRST
            if k not in seen_tv:
                seen_tv.add(k)
                while len(hdr.tyvars) <= k: hdr.tyvars.append(f"a{len(hdr.tyvars)}")
    return hdr

def has_step_block(toks):
    """True iff `(KW_step` appears in the token list."""
    for i, t in enumerate(toks):
        if t == T.KW_STEP and i > 0 and toks[i - 1] == T.LPAREN:
            return True
    return False

def _canonicalize_term(term):
    """Rename free variables to vc0, vc1, … in order of first appearance.
    Bound variables keep their original names (alpha_eq handles those via
    binder-depth indexing).  This lets us compare two terms that are
    structurally identical but use different free-var names."""
    rename = {}
    def aux(t, bound):
        if t[0] == "Var":
            n, ty = t[1], t[2]
            if n in bound:
                return t
            if n not in rename:
                rename[n] = f"vc{len(rename)}"
            return ("Var", rename[n], ty)
        if t[0] == "Const":
            return t
        if t[0] == "Comb":
            return ("Comb", aux(t[1], bound), aux(t[2], bound))
        if t[0] == "Abs":
            return ("Abs", t[1], t[2], aux(t[3], bound | {t[1]}))
        return t
    return aux(term, set())

def _kernel_verify(cert_toks, goal_toks):
    """Send cert+goal tokens to the OCaml verifier subprocess; read reward."""
    if _W_verifier is None: return -1
    parts = [str(len(cert_toks))] + [str(t) for t in cert_toks]
    parts += [str(len(goal_toks))] + [str(t) for t in goal_toks]
    req = " ".join(parts) + "\n"
    try:
        _W_verifier.stdin.write(req)
        _W_verifier.stdin.flush()
        line = _W_verifier.stdout.readline()
        if not line: return -1
        return int(line.strip())
    except Exception:
        return -1

def V_python(goal, gen_tokens):
    """Pure-Python V (kept as fallback when kernel verifier unavailable).
    Returns -1 on grammar reject, REWARD_WRONG_CONCL on bad concl, +100 on match,
    plus an optional STEP_BONUS if the cert has a step block."""
    s = T.initial_state()
    for t in gen_tokens:
        ns = T.step(s, t)
        if ns is None: return -1.0
        s = ns
    if not T.is_accepting(s):
        return -1.0
    bonus = STEP_BONUS if has_step_block(gen_tokens) else 0.0
    try:
        hdr = _infer_header(gen_tokens)
        cert = T.decode_cert(hdr, gen_tokens)
    except Exception:
        return REWARD_WRONG_CONCL + bonus
    base = 100.0 if T.alpha_eq(cert.concl, goal) else REWARD_WRONG_CONCL
    return base + bonus

def V(goal, goal_toks, gen_tokens):
    """Score a completion.  Uses the OCaml kernel verifier when available
    (kills the zero-step cheat), else the Python fallback."""
    if USE_KERNEL_VERIFY and _W_verifier is not None:
        return float(_kernel_verify(gen_tokens, goal_toks))
    return V_python(goal, gen_tokens)

# ---------------------------------------------------------------------------
# Worker entry-points.
# ---------------------------------------------------------------------------
def _worker_init():
    global _W_state_dict, _W_params, _W_n_params, _W_verifier
    _W_state_dict, _W_params = _build_model()
    _W_n_params = len(_W_params)
    if USE_KERNEL_VERIFY and os.path.exists(VERIFIER_BIN):
        try:
            _W_verifier = subprocess.Popen(
                [VERIFIER_BIN],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=1, universal_newlines=True,
            )
        except Exception:
            _W_verifier = None

def _apply_params_data(params_data):
    for p, d in zip(_W_params, params_data):
        p.data = d
        p.grad = 0

def _worker_rollout(args):
    """Run one rollout.  Returns (grads_of_full_loss, reward, gen_len).

    The full loss combines policy-gradient and entropy regularisation:
        loss = -advantage * Σ log π(a_t | s_<t)  -  β · Σ H(π_t)
    Worker computes its own advantage from `baseline` (passed in) and runs
    a single backward.  Main just averages the resulting gradients."""
    params_data, goal, goal_toks, baseline, rng_seed = args
    _apply_params_data(params_data)
    random.seed(rng_seed)
    prompt = [BOS] + list(goal_toks) + [BOS]
    if len(prompt) >= block_size - 4:
        return [0.0] * _W_n_params, -1.0, 0
    keys = [[] for _ in range(n_layer)]; values = [[] for _ in range(n_layer)]
    logits = None
    for pos_id, tok in enumerate(prompt):
        logits = _gpt(_W_state_dict, tok, pos_id, keys, values)
    grammar_state = T.initial_state()
    log_probs, entropies, generated = [], [], []
    pos, gen_count = len(prompt), 0
    while pos < block_size and gen_count < max_gen_tokens:
        mask = T.valid_next_mask(grammar_state)
        sampled, lp, ent = sample_with_mask(logits, mask)
        log_probs.append(lp); entropies.append(ent); generated.append(sampled)
        ns = T.step(grammar_state, sampled)
        if ns is None: break
        grammar_state = ns
        gen_count += 1
        if T.is_accepting(grammar_state): break
        if pos >= block_size - 1 or gen_count >= max_gen_tokens: break
        logits = _gpt(_W_state_dict, sampled, pos, keys, values)
        pos += 1
    reward = V(goal, goal_toks, generated)
    if log_probs:
        advantage = reward - baseline
        # Apply entropy bonus only when this rollout didn't succeed.  This
        # keeps solved rollouts (R≈+100) from drifting via entropy pressure
        # — only the rollouts that need exploration carry the H term.
        eff_beta = ENTROPY_BETA if reward < 50.0 else 0.0
        loss = (-advantage) * sum(log_probs) + (-eff_beta) * sum(entropies)
        loss.backward()
        grads = [p.grad for p in _W_params]
    else:
        grads = [0.0] * _W_n_params
    return grads, reward, len(generated)

def _worker_warmup(args):
    """Teacher-forced CE on (prompt + gold cert).  Returns grads, avg log-prob."""
    params_data, goal_toks, cert_toks = args
    _apply_params_data(params_data)
    prompt = [BOS] + list(goal_toks) + [BOS]
    target = list(cert_toks) + [EOS]
    full_seq = prompt + target
    if len(full_seq) >= block_size:
        return [0.0] * _W_n_params, 0.0
    keys = [[] for _ in range(n_layer)]; values = [[] for _ in range(n_layer)]
    total_log_p = Value(0.0)
    n_terms = 0
    # We want to predict full_seq[i+1] given full_seq[:i+1].  Loss is over
    # the completion portion (positions >= len(prompt) - 1).
    for pos in range(len(full_seq) - 1):
        logits = _gpt(_W_state_dict, full_seq[pos], pos, keys, values)
        nxt = full_seq[pos + 1]
        if pos < len(prompt) - 1:
            continue
        probs = _softmax_data(logits)
        total_log_p = total_log_p + probs[nxt].log()
        n_terms += 1
    if n_terms == 0:
        return [0.0] * _W_n_params, 0.0
    loss = -total_log_p
    loss.backward()
    grads = [p.grad for p in _W_params]
    return grads, total_log_p.data / max(1, n_terms)

# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    log(f"=== microgpt_with_RL_hol starting (pid={os.getpid()}) ===")
    log(f"NUM_STEPS={NUM_STEPS}  LOG_EVERY={LOG_EVERY}  CKPT_EVERY={CKPT_EVERY}")
    log(f"N_WORKERS={N_WORKERS}  WARMUP_STEPS={WARMUP_STEPS}")
    log(f"REWARD_WRONG_CONCL={REWARD_WRONG_CONCL}  STEP_BONUS={STEP_BONUS}")
    log(f"vocab_size={vocab_size}  block_size={block_size}  max_gen_tokens={max_gen_tokens}")
    log(f"#seeds: {len(SEEDS)}")
    for label, _, gt, ct in encoded_goals:
        log(f"  {label}: prompt-toks={len(gt)} gold-cert-toks={len(ct)}")

    state_dict, params = _build_model()
    # Re-init weights with proper distribution (workers do the same).
    rng = random.Random(42)
    for mat in state_dict.values():
        for row in mat:
            for v in row: v.data = rng.gauss(0, 0.08)
    log(f"num params: {len(params)}")

    # Adam state.
    learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
    m_state  = [0.0] * len(params)
    v_state  = [0.0] * len(params)
    start_step = 0
    baselines = {label: 0.0 for (label, _, _, _) in encoded_goals}
    baseline_ema = 0.9

    # Resume from checkpoint if present.
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH, 'rb') as f:
            ckpt = pickle.load(f)
        for p, d in zip(params, ckpt['params_data']):
            p.data = d
        m_state[:] = ckpt['m']
        v_state[:] = ckpt['v_buf']
        baselines.update(ckpt['baselines'])
        start_step = ckpt['step']
        log(f"resumed from checkpoint at step {start_step}")

    def snapshot():
        return [p.data for p in params]

    def save_ckpt(step):
        ckpt = {
            'step': step,
            'params_data': snapshot(),
            'm': m_state, 'v_buf': v_state,
            'baselines': baselines,
        }
        with open(CKPT_PATH, 'wb') as f:
            pickle.dump(ckpt, f)
        log(f"checkpoint saved at step {step}")

    def adam_step(grad, step, total=None):
        # LR decays linearly to 0 over `total` steps; default to NUM_STEPS.
        total = total if total is not None else NUM_STEPS
        lr_t = learning_rate * max(0.0, 1 - step / max(1, total))
        for i, p in enumerate(params):
            m_state[i] = beta1 * m_state[i] + (1 - beta1) * grad[i]
            v_state[i] = beta2 * v_state[i] + (1 - beta2) * grad[i] ** 2
            m_hat = m_state[i] / (1 - beta1 ** (step + 1))
            v_hat = v_state[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)

    log(f"creating pool of {N_WORKERS} workers")
    pool = mp.Pool(N_WORKERS, initializer=_worker_init)

    # ----- Warmup (supervised, teacher-forced) -----------------------------
    if WARMUP_STEPS > 0 and start_step == 0:
        log(f"=== Warmup: {WARMUP_STEPS} supervised steps over "
            f"{len(ENCODED_CORPUS)} corpus examples ===")
        t_start = time.time()
        rng_warm = random.Random(13)
        for w_step in range(WARMUP_STEPS):
            params_data = snapshot()
            # Sample N_WORKERS distinct examples from the corpus (with
            # replacement if the corpus is smaller than N_WORKERS).
            n_samp = min(N_WORKERS, len(ENCODED_CORPUS))
            picks = rng_warm.sample(range(len(ENCODED_CORPUS)), n_samp) \
                    if len(ENCODED_CORPUS) >= n_samp else \
                    [rng_warm.randrange(len(ENCODED_CORPUS)) for _ in range(N_WORKERS)]
            # Pad up to N_WORKERS by re-sampling with replacement.
            while len(picks) < N_WORKERS:
                picks.append(rng_warm.randrange(len(ENCODED_CORPUS)))
            args = [(params_data, ENCODED_CORPUS[i][0], ENCODED_CORPUS[i][1])
                    for i in picks]
            results = pool.map(_worker_warmup, args)
            avg = [0.0] * len(params)
            for grads, _ in results:
                for i, g in enumerate(grads):
                    avg[i] += g
            avg = [g / len(results) for g in avg]
            # Use a large `total` so warmup runs at near-constant LR.
            adam_step(avg, w_step, total=max(NUM_STEPS, WARMUP_STEPS * 50))
            if (w_step + 1) % max(1, WARMUP_STEPS // 20) == 0:
                avg_lp = sum(lp for _, lp in results) / len(results)
                log(f"  warmup {w_step+1:4d}/{WARMUP_STEPS}  avg-log-p={avg_lp:+.3f}")
        log(f"=== Warmup done ({time.time()-t_start:.1f}s) ===")

    # ----- RL loop ---------------------------------------------------------
    t_start = time.time()
    last_log_t = t_start
    for step in range(start_step, NUM_STEPS):
        label, goal, goal_toks, _cert_toks = encoded_goals[step % len(encoded_goals)]
        params_data = snapshot()
        b = baselines[label]
        args = [(params_data, goal, goal_toks, b, step * N_WORKERS + i) for i in range(N_WORKERS)]
        results = pool.map(_worker_rollout, args)

        rewards = [r for _, r, _ in results]
        gen_lens = [gl for _, _, gl in results]
        avg_r = sum(rewards) / len(rewards)

        # Workers already scaled grads by -(reward - baseline) and added the
        # -β·∂H term.  We just average across workers.
        n = len(results)
        avg_grad = [0.0] * len(params)
        for (grads, _, _) in results:
            for i, g in enumerate(grads):
                avg_grad[i] += g / n
        adam_step(avg_grad, step)

        # Update baseline with the average reward across this step's workers.
        baselines[label] = baseline_ema * b + (1 - baseline_ema) * avg_r

        if (step + 1) % LOG_EVERY == 0:
            now = time.time()
            rate = LOG_EVERY / (now - last_log_t) if now > last_log_t else 0
            last_log_t = now
            r_min, r_max = min(rewards), max(rewards)
            best_gen = gen_lens[rewards.index(r_max)]
            avg_b = sum(baselines.values()) / len(baselines)
            log(f"step {step+1:5d}/{NUM_STEPS} | {label:14s} | "
                f"R avg {avg_r:+7.2f} (min {r_min:+6.1f}, max {r_max:+6.1f}) | "
                f"best-gen-len {best_gen:3d} | avg-b {avg_b:+7.2f} | "
                f"{rate:.2f} steps/s")

        if (step + 1) % CKPT_EVERY == 0:
            save_ckpt(step + 1)

    # ----- Inference -------------------------------------------------------
    log("=== Inference (greedy) ===")
    for label, goal, goal_toks, _ct in encoded_goals:
        prompt = [BOS] + list(goal_toks) + [BOS]
        keys = [[] for _ in range(n_layer)]; values = [[] for _ in range(n_layer)]
        logits = None
        for pos_id, tok in enumerate(prompt):
            logits = _gpt(state_dict, tok, pos_id, keys, values)
        grammar_state = T.initial_state()
        out = []
        pos, gen_count = len(prompt), 0
        while pos < block_size and gen_count < max_gen_tokens:
            mask = T.valid_next_mask(grammar_state)
            sampled = argmax_with_mask(logits, mask)
            if sampled is None: break
            out.append(sampled)
            ns = T.step(grammar_state, sampled)
            if ns is None: break
            grammar_state = ns
            gen_count += 1
            if T.is_accepting(grammar_state): break
            if pos >= block_size - 1 or gen_count >= max_gen_tokens: break
            logits = _gpt(state_dict, sampled, pos, keys, values)
            pos += 1
        # Inference runs in main; no worker subprocess.  Fall back to
        # Python V which doesn't use kernel-verify.  Good enough for an
        # end-of-run summary; the training reward is what we care about.
        r = V_python(goal, out)
        log(f"  inference {label:14s} gen-len {len(out):3d} V={r:+.1f}")

    save_ckpt(NUM_STEPS)
    log("=== run complete ===")
    pool.close(); pool.join()
    log_f.close()

if __name__ == "__main__":
    main()

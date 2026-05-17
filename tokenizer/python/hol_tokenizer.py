"""hol_tokenizer.py — Python port of the OCaml HOL certificate tokenizer.

Mirrors:
- tokenizer/lexicon.ml  — vocab table, token IDs
- tokenizer/encode.ml   — Cert -> token-ID list + pool header
- tokenizer/decode.ml   — pool header + tokens -> Cert
- tokenizer/grammar.ml  — pushdown automaton; valid_next_mask

Designed to feed a vanilla GPT (e.g. Karpathy's microgpt) on lexeme-level
token IDs in [0 .. vocab_size).  The grammar mask lets the model do
constrained decoding so every sampled completion is a syntactically
well-formed cert prefix.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Sequence

# ---------------------------------------------------------------------------
# Lexicon: token IDs (matches lexicon.ml byte-for-byte).
# ---------------------------------------------------------------------------

BOS, EOS, PAD, UNK = 0, 1, 2, 3

LPAREN = 4
RPAREN = 5
COLON = 6
DOT = 7
COMMA = 8
QUOTE = 9
ARROW = 10

OP_EQ = 16
OP_IMP = 17
OP_CONJ = 18
OP_DISJ = 19
OP_NOT = 20
OP_FORALL = 21
OP_EXISTS = 22
OP_LAMBDA = 23

KW_TYPE = 24
KW_CONST = 25
KW_AXIOM = 26
KW_THEOREM = 27
KW_GOAL = 28
KW_CERT = 29
KW_STEP = 30
KW_RULE = 31
KW_WITNESS = 32
KW_PREMISES = 33
KW_CONCL = 34
KW_TERM = 35
KW_VAR = 36
KW_INST = 37
KW_INSTTYPE = 38
KW_B_AND_W = 39
KW_SUBST = 40
KW_BOUND = 41

RULE_FIRST = 48
RULE_NAMES = [
    "REFL", "TRANS", "MK_COMB", "ABS", "BETA", "ASSUME", "EQ_MP",
    "DEDUCT_ANTISYM_RULE", "INST", "INST_TYPE", "AXIOM", "GEN", "SPEC",
    "EXISTS", "CHOOSE", "CONJ", "CONJUNCT1", "CONJUNCT2", "MP", "DISCH",
    "ETA_AX", "SELECT_AX", "EM_AX"
]
RULE_TOK = {n: RULE_FIRST + i for i, n in enumerate(RULE_NAMES)}
RULE_FROM_TOK = {RULE_FIRST + i: n for i, n in enumerate(RULE_NAMES)}
RULE_LAST = RULE_FIRST + len(RULE_NAMES) - 1

TY_BOOL = 72
TY_IND = 73
TY_FUN = 74
TY_NAT = 75
BUILTIN_TY = {"bool": TY_BOOL, "ind": TY_IND, "fun": TY_FUN, "nat": TY_NAT}
BUILTIN_FROM_TOK = {v: k for k, v in BUILTIN_TY.items()}

TYVAR_FIRST = 80
TYVAR_COUNT = 8
TYVAR_LAST = TYVAR_FIRST + TYVAR_COUNT - 1

TYCON_FIRST = 88
TYCON_COUNT = 16
TYCON_LAST = TYCON_FIRST + TYCON_COUNT - 1

NAME_FIRST = 104
NAME_COUNT = 64
NAME_LAST = NAME_FIRST + NAME_COUNT - 1

VAR_FIRST = 168
VAR_COUNT = 32
VAR_LAST = VAR_FIRST + VAR_COUNT - 1

INT_FIRST = 200
INT_COUNT = 32
INT_LAST = INT_FIRST + INT_COUNT - 1

VOCAB_SIZE = 232


def is_rule(t):    return RULE_FIRST <= t <= RULE_LAST
def is_tyvar(t):   return TYVAR_FIRST <= t <= TYVAR_LAST
def is_tycon(t):   return TYCON_FIRST <= t <= TYCON_LAST
def is_name(t):    return NAME_FIRST <= t <= NAME_LAST
def is_var(t):     return VAR_FIRST <= t <= VAR_LAST
def is_int_tok(t): return INT_FIRST <= t <= INT_LAST
def is_builtin_ty(t): return t in BUILTIN_FROM_TOK
def is_ty_head(t): return is_builtin_ty(t) or is_tycon(t) or is_tyvar(t)


def tok_str(t):
    """Human-readable rendering of a token ID, for debugging."""
    if t == BOS: return "BOS"
    if t == EOS: return "EOS"
    if t == PAD: return "PAD"
    if t == UNK: return "UNK"
    if t == LPAREN: return "("
    if t == RPAREN: return ")"
    if t == COLON: return ":"
    if t == DOT: return "."
    if t == COMMA: return ","
    if t == QUOTE: return '"'
    if t == ARROW: return "->"
    if t == OP_EQ: return "="
    if t == OP_IMP: return "==>"
    if t == OP_CONJ: return "/\\"
    if t == OP_DISJ: return "\\/"
    if t == OP_NOT: return "~"
    if t == OP_FORALL: return "!"
    if t == OP_EXISTS: return "?"
    if t == OP_LAMBDA: return "\\"
    if t == KW_TYPE: return "KW_type"
    if t == KW_CONST: return "KW_const"
    if t == KW_AXIOM: return "KW_axiom"
    if t == KW_THEOREM: return "KW_theorem"
    if t == KW_GOAL: return "KW_goal"
    if t == KW_CERT: return "KW_cert"
    if t == KW_STEP: return "KW_step"
    if t == KW_RULE: return "KW_rule"
    if t == KW_WITNESS: return "KW_witness"
    if t == KW_PREMISES: return "KW_premises"
    if t == KW_CONCL: return "KW_concl"
    if t == KW_TERM: return "KW_term"
    if t == KW_VAR: return "KW_var"
    if t == KW_INST: return "KW_inst"
    if t == KW_INSTTYPE: return "KW_insttype"
    if t == KW_B_AND_W: return "KW_b_and_w"
    if t == KW_SUBST: return "KW_subst"
    if t == KW_BOUND: return "KW_bound"
    if t in RULE_FROM_TOK: return "RULE_" + RULE_FROM_TOK[t]
    if t in BUILTIN_FROM_TOK: return "TY_" + BUILTIN_FROM_TOK[t]
    if is_tyvar(t): return f"TYVAR_a{t - TYVAR_FIRST}"
    if is_tycon(t): return f"TYCON_t{t - TYCON_FIRST}"
    if is_name(t):  return f"NAME_n{t - NAME_FIRST}"
    if is_var(t):   return f"VAR_v{t - VAR_FIRST}"
    if is_int_tok(t): return f"INT_{t - INT_FIRST}"
    return f"?<{t}>"


# ---------------------------------------------------------------------------
# HOL term / type / cert representation.
# ---------------------------------------------------------------------------

# Type:  ("Tyvar", str)            -> a type variable
#        ("Tyapp", str, [Type])    -> a type-constructor application
#
# Term:  ("Var", str, Type)
#        ("Const", str, Type)
#        ("Comb", Term, Term)
#        ("Abs", str, Type, Term)  -- binder name, binder type, body

def tyvar(name):       return ("Tyvar", name)
def tyapp(name, args): return ("Tyapp", name, list(args))
def bool_ty():         return tyapp("bool", [])
def ind_ty():          return tyapp("ind",  [])
def fun_ty(a, b):      return tyapp("fun",  [a, b])
def nat_ty():          return tyapp("nat",  [])


def type_equal(a, b):
    if a[0] != b[0]: return False
    if a[0] == "Tyvar":
        return a[1] == b[1]
    return (a[1] == b[1]
            and len(a[2]) == len(b[2])
            and all(type_equal(x, y) for x, y in zip(a[2], b[2])))


def mk_var(name, ty):   return ("Var", name, ty)
def mk_const(name, ty): return ("Const", name, ty)
def mk_comb(f, x):      return ("Comb", f, x)
def mk_abs(n, ty, b):   return ("Abs", n, ty, b)


def mk_eq(s, t):
    # Polymorphic '=' const at the type of s.  type_of(t) must equal type_of(s).
    ty = type_of(s)
    eq_ty = fun_ty(ty, fun_ty(ty, bool_ty()))
    return mk_comb(mk_comb(mk_const("=", eq_ty), s), t)


def mk_imp(p, q):
    bb_b = fun_ty(bool_ty(), fun_ty(bool_ty(), bool_ty()))
    return mk_comb(mk_comb(mk_const("==>", bb_b), p), q)


def mk_conj(p, q):
    bb_b = fun_ty(bool_ty(), fun_ty(bool_ty(), bool_ty()))
    return mk_comb(mk_comb(mk_const("/\\", bb_b), p), q)


def mk_disj(p, q):
    bb_b = fun_ty(bool_ty(), fun_ty(bool_ty(), bool_ty()))
    return mk_comb(mk_comb(mk_const("\\/", bb_b), p), q)


def mk_not(p):
    b_b = fun_ty(bool_ty(), bool_ty())
    return mk_comb(mk_const("~", b_b), p)


def mk_forall(name, ty, body):
    pty = fun_ty(ty, bool_ty())
    qty = fun_ty(pty, bool_ty())
    return mk_comb(mk_const("!", qty), mk_abs(name, ty, body))


def mk_exists(name, ty, body):
    pty = fun_ty(ty, bool_ty())
    qty = fun_ty(pty, bool_ty())
    return mk_comb(mk_const("?", qty), mk_abs(name, ty, body))


def type_of(t):
    if t[0] == "Var" or t[0] == "Const": return t[2]
    if t[0] == "Comb":
        ft = type_of(t[1])
        assert ft[0] == "Tyapp" and ft[1] == "fun", f"type_of comb: lhs not function {ft}"
        return ft[2][1]
    if t[0] == "Abs":
        return fun_ty(t[2], type_of(t[3]))
    raise ValueError(f"type_of: {t}")


def alpha_eq(t1, t2, env1=None, env2=None):
    if env1 is None: env1, env2 = [], []
    if t1[0] != t2[0]: return False
    if t1[0] == "Var":
        a, ta = t1[1], t1[2]
        b, tb = t2[1], t2[2]
        if not type_equal(ta, tb): return False
        i = next((d for (n, d) in env1 if n == a), None)
        j = next((d for (n, d) in env2 if n == b), None)
        if i is not None and j is not None: return i == j
        if i is None and j is None:         return a == b
        return False
    if t1[0] == "Const":
        return t1[1] == t2[1] and type_equal(t1[2], t2[2])
    if t1[0] == "Comb":
        return alpha_eq(t1[1], t2[1], env1, env2) and alpha_eq(t1[2], t2[2], env1, env2)
    if t1[0] == "Abs":
        if not type_equal(t1[2], t2[2]): return False
        d = len(env1)
        return alpha_eq(t1[3], t2[3], [(t1[1], d)] + env1, [(t2[1], d)] + env2)
    return False


# A certificate.
@dataclass
class Step:
    id: int
    rule: str
    witness: tuple                 # ("none",) | ("term", term) | ("type", ty) | ("var", name, ty) | ("axiom", name) | ("inst", [(v, t)]) | ("insttype", [(name, ty)]) | ("bnw", name, ty, term)
    premises: List[int]

@dataclass
class Cert:
    steps: List[Step]
    concl: tuple


# ---------------------------------------------------------------------------
# Pool header for round-trip.
# ---------------------------------------------------------------------------

@dataclass
class PoolHeader:
    tycons: List[str] = field(default_factory=list)
    tyvars: List[str] = field(default_factory=list)
    names:  List[str] = field(default_factory=list)
    vars:   List[str] = field(default_factory=list)


class _EncCtx:
    def __init__(self):
        self.tycons, self.tyvars, self.names, self.vars = [], [], [], []
        self.out: List[int] = []

    def emit(self, t): self.out.append(t)

    def _alloc(self, lst, limit, kind, name):
        try:
            return lst.index(name)
        except ValueError:
            if len(lst) >= limit:
                raise RuntimeError(f"{kind} pool full at {name}")
            lst.append(name)
            return len(lst) - 1

    def alloc_tycon(self, n): return self._alloc(self.tycons, TYCON_COUNT, "tycon", n)
    def alloc_tyvar(self, n): return self._alloc(self.tyvars, TYVAR_COUNT, "tyvar", n)
    def alloc_name(self, n):  return self._alloc(self.names,  NAME_COUNT,  "name",  n)
    def alloc_var(self, n):   return self._alloc(self.vars,   VAR_COUNT,   "var",   n)


# ---------------------------------------------------------------------------
# Encoder.
# ---------------------------------------------------------------------------

def _encode_type(ctx, ty):
    if ty[0] == "Tyvar":
        ctx.emit(TYVAR_FIRST + ctx.alloc_tyvar(ty[1]))
    else:
        name, args = ty[1], ty[2]
        if name == "fun" and len(args) == 2:
            ctx.emit(LPAREN); _encode_type(ctx, args[0]); ctx.emit(ARROW)
            _encode_type(ctx, args[1]); ctx.emit(RPAREN)
        elif len(args) == 0:
            if name in BUILTIN_TY:
                ctx.emit(BUILTIN_TY[name])
            else:
                ctx.emit(TYCON_FIRST + ctx.alloc_tycon(name))
        else:
            head = BUILTIN_TY[name] if name in BUILTIN_TY else (TYCON_FIRST + ctx.alloc_tycon(name))
            ctx.emit(LPAREN)
            for i, a in enumerate(args):
                if i: ctx.emit(COMMA)
                _encode_type(ctx, a)
            ctx.emit(RPAREN)
            ctx.emit(head)


def _emit_var_ref(ctx, name):
    ctx.emit(VAR_FIRST + ctx.alloc_var(name))


def _emit_const_ref(ctx, name):
    inline = {"=": OP_EQ, "/\\": OP_CONJ, "\\/": OP_DISJ,
              "==>": OP_IMP, "~": OP_NOT, "!": OP_FORALL, "?": OP_EXISTS}
    if name in inline:
        ctx.emit(inline[name])
    else:
        ctx.emit(NAME_FIRST + ctx.alloc_name(name))


def _encode_term(ctx, t):
    # Binary connective patterns first
    if t[0] == "Comb":
        f, x = t[1], t[2]
        if f[0] == "Comb":
            g = f[1]
            if g[0] == "Const":
                cn = g[1]
                if cn in ("=", "/\\", "\\/", "==>"):
                    inner = f[2]
                    op_map = {"=": OP_EQ, "/\\": OP_CONJ, "\\/": OP_DISJ, "==>": OP_IMP}
                    ctx.emit(LPAREN); _encode_term(ctx, inner); ctx.emit(op_map[cn])
                    _encode_term(ctx, x); ctx.emit(RPAREN); return
        if f[0] == "Const" and f[1] == "~":
            ctx.emit(LPAREN); ctx.emit(OP_NOT); _encode_term(ctx, x); ctx.emit(RPAREN); return
        if f[0] == "Const" and f[1] == "!" and x[0] == "Abs":
            _encode_binder(ctx, OP_FORALL, x); return
        if f[0] == "Const" and f[1] == "?" and x[0] == "Abs":
            _encode_binder(ctx, OP_EXISTS, x); return
        # Generic application: ( f x )
        ctx.emit(LPAREN); _encode_term(ctx, f); _encode_term(ctx, x); ctx.emit(RPAREN); return
    if t[0] == "Abs":
        _encode_binder(ctx, OP_LAMBDA, t); return
    if t[0] == "Var":
        _emit_var_ref(ctx, t[1])
        ctx.emit(COLON); _encode_type(ctx, t[2]); return
    if t[0] == "Const":
        _emit_const_ref(ctx, t[1]); return
    raise ValueError(f"_encode_term: {t}")


def _encode_binder(ctx, op_tok, abs_term):
    # abs_term = ("Abs", name, ty, body)
    n, ty, body = abs_term[1], abs_term[2], abs_term[3]
    ctx.emit(LPAREN); ctx.emit(op_tok)
    _emit_var_ref(ctx, n); ctx.emit(COLON); _encode_type(ctx, ty); ctx.emit(DOT)
    _encode_term(ctx, body); ctx.emit(RPAREN)


def _encode_int(ctx, n):
    if n < 0 or n >= INT_COUNT:
        raise RuntimeError(f"INT literal out of range: {n}")
    ctx.emit(INT_FIRST + n)


def _encode_witness(ctx, w):
    if w[0] == "none":
        return
    ctx.emit(LPAREN)
    tag = w[0]
    if tag == "term":
        ctx.emit(KW_TERM); ctx.emit(QUOTE); _encode_term(ctx, w[1]); ctx.emit(QUOTE)
    elif tag == "type":
        ctx.emit(KW_TYPE); ctx.emit(QUOTE); _encode_type(ctx, w[1]); ctx.emit(QUOTE)
    elif tag == "var":
        ctx.emit(KW_VAR); ctx.emit(QUOTE); _emit_var_ref(ctx, w[1]); ctx.emit(QUOTE)
        ctx.emit(QUOTE); _encode_type(ctx, w[2]); ctx.emit(QUOTE)
    elif tag == "axiom":
        ctx.emit(KW_AXIOM); ctx.emit(QUOTE)
        ctx.emit(NAME_FIRST + ctx.alloc_name(w[1])); ctx.emit(QUOTE)
    elif tag == "inst":
        ctx.emit(KW_INST)
        for v, rhs in w[1]:
            ctx.emit(LPAREN); ctx.emit(KW_SUBST)
            ctx.emit(QUOTE); _encode_term(ctx, v); ctx.emit(QUOTE)
            ctx.emit(QUOTE); _encode_term(ctx, rhs); ctx.emit(QUOTE)
            ctx.emit(RPAREN)
    elif tag == "insttype":
        ctx.emit(KW_INSTTYPE)
        for a, ty in w[1]:
            ctx.emit(LPAREN); ctx.emit(KW_SUBST)
            ctx.emit(QUOTE); _emit_var_ref(ctx, a); ctx.emit(QUOTE)
            ctx.emit(QUOTE); _encode_type(ctx, ty); ctx.emit(QUOTE)
            ctx.emit(RPAREN)
    elif tag == "bnw":
        ctx.emit(KW_B_AND_W)
        ctx.emit(LPAREN); ctx.emit(KW_BOUND)
        ctx.emit(QUOTE); _emit_var_ref(ctx, w[1]); ctx.emit(QUOTE)
        ctx.emit(QUOTE); _encode_type(ctx, w[2]); ctx.emit(QUOTE)
        ctx.emit(RPAREN)
        ctx.emit(LPAREN); ctx.emit(KW_WITNESS)
        ctx.emit(QUOTE); _encode_term(ctx, w[3]); ctx.emit(QUOTE)
        ctx.emit(RPAREN)
    else:
        raise ValueError(f"unknown witness tag: {tag}")
    ctx.emit(RPAREN)


def _encode_step(ctx, s: Step):
    ctx.emit(LPAREN); ctx.emit(KW_STEP)
    _encode_int(ctx, s.id)
    ctx.emit(LPAREN); ctx.emit(KW_RULE)
    if s.rule not in RULE_TOK: raise RuntimeError(f"unknown rule: {s.rule}")
    ctx.emit(RULE_TOK[s.rule]); ctx.emit(RPAREN)
    ctx.emit(LPAREN); ctx.emit(KW_WITNESS); _encode_witness(ctx, s.witness); ctx.emit(RPAREN)
    ctx.emit(LPAREN); ctx.emit(KW_PREMISES)
    for p in s.premises: _encode_int(ctx, p)
    ctx.emit(RPAREN); ctx.emit(RPAREN)


def encode_cert(cert: Cert) -> Tuple[List[int], PoolHeader]:
    ctx = _EncCtx()
    ctx.emit(LPAREN); ctx.emit(KW_CERT)
    for s in cert.steps: _encode_step(ctx, s)
    ctx.emit(LPAREN); ctx.emit(KW_CONCL)
    ctx.emit(QUOTE); _encode_term(ctx, cert.concl); ctx.emit(QUOTE)
    ctx.emit(RPAREN); ctx.emit(RPAREN)
    return ctx.out, PoolHeader(ctx.tycons[:], ctx.tyvars[:], ctx.names[:], ctx.vars[:])


def encode_term_only(term, ctx=None) -> Tuple[List[int], PoolHeader]:
    """Encode a HOL term as a token sequence (no surrounding cert/quote
    wrapping).  Useful for prompt strings — a theorem stated alone."""
    ctx = ctx or _EncCtx()
    _encode_term(ctx, term)
    return ctx.out, PoolHeader(ctx.tycons[:], ctx.tyvars[:], ctx.names[:], ctx.vars[:])


# ---------------------------------------------------------------------------
# Decoder.
# ---------------------------------------------------------------------------

class _Cur:
    def __init__(self, toks):
        self.toks = list(toks); self.pos = 0
    def peek(self):
        if self.pos >= len(self.toks):
            raise RuntimeError("decode: unexpected EOF")
        return self.toks[self.pos]
    def advance(self): self.pos += 1
    def expect(self, t):
        if self.peek() != t:
            raise RuntimeError(f"decode: expected {tok_str(t)} got {tok_str(self.peek())} @ {self.pos}")
        self.advance()


def _var_name_in(hdr: PoolHeader, k):
    return hdr.vars[k] if k < len(hdr.vars) else f"v{k}"


def _const_type(name):
    # Minimal table for connectives + Euclid constants.  Add more as needed.
    bool_b   = fun_ty(bool_ty(), bool_ty())
    bool_bb  = fun_ty(bool_ty(), fun_ty(bool_ty(), bool_ty()))
    A = tyvar("A")
    poly_eq  = fun_ty(A, fun_ty(A, bool_ty()))
    builtin = {
        "=":    poly_eq,
        "==>":  bool_bb,
        "/\\":  bool_bb,
        "\\/":  bool_bb,
        "~":    bool_b,
        "!":    fun_ty(fun_ty(A, bool_ty()), bool_ty()),
        "?":    fun_ty(fun_ty(A, bool_ty()), bool_ty()),
        "T":    bool_ty(),
        "F":    bool_ty(),
    }
    if name in builtin: return builtin[name]
    return None  # caller must supply or fail


_extra_const_types = {}

def register_const(name, ty):
    """Register a theory-scoped constant's type so the decoder can
    reconstruct Const(name, ty)."""
    _extra_const_types[name] = ty


def _decode_type(hdr, c):
    tok = c.peek()
    if tok == LPAREN:
        c.advance()
        first = _decode_type(hdr, c)
        nxt = c.peek()
        if nxt == ARROW:
            c.advance(); rhs = _decode_type(hdr, c); c.expect(RPAREN)
            return fun_ty(first, rhs)
        args = [first]
        while c.peek() == COMMA:
            c.advance(); args.append(_decode_type(hdr, c))
        c.expect(RPAREN)
        head = c.peek()
        if head in BUILTIN_FROM_TOK:
            name = BUILTIN_FROM_TOK[head]; c.advance()
        elif is_tycon(head):
            name = hdr.tycons[head - TYCON_FIRST]; c.advance()
        else:
            raise RuntimeError(f"decode_type: expected tycon head got {tok_str(head)}")
        return tyapp(name, args)
    if is_tyvar(tok):
        c.advance(); return tyvar(hdr.tyvars[tok - TYVAR_FIRST])
    if tok in BUILTIN_FROM_TOK:
        c.advance(); return tyapp(BUILTIN_FROM_TOK[tok], [])
    if is_tycon(tok):
        c.advance(); return tyapp(hdr.tycons[tok - TYCON_FIRST], [])
    raise RuntimeError(f"decode_type: unexpected {tok_str(tok)}")


def _decode_term(hdr, c):
    tok = c.peek()
    if tok == LPAREN:
        c.advance()
        nxt = c.peek()
        if nxt in (OP_FORALL, OP_EXISTS, OP_LAMBDA):
            c.advance()
            vt = c.peek();
            if not is_var(vt): raise RuntimeError("binder: expected var")
            c.advance(); c.expect(COLON)
            ty = _decode_type(hdr, c); c.expect(DOT)
            body = _decode_term(hdr, c); c.expect(RPAREN)
            n = _var_name_in(hdr, vt - VAR_FIRST)
            if nxt == OP_FORALL: return mk_forall(n, ty, body)
            if nxt == OP_EXISTS: return mk_exists(n, ty, body)
            return mk_abs(n, ty, body)
        if nxt == OP_NOT:
            c.advance(); body = _decode_term(hdr, c); c.expect(RPAREN)
            return mk_not(body)
        t1 = _decode_term(hdr, c)
        op = c.peek()
        if op == OP_EQ:    c.advance(); t2 = _decode_term(hdr, c); c.expect(RPAREN); return mk_eq(t1, t2)
        if op == OP_CONJ:  c.advance(); t2 = _decode_term(hdr, c); c.expect(RPAREN); return mk_conj(t1, t2)
        if op == OP_DISJ:  c.advance(); t2 = _decode_term(hdr, c); c.expect(RPAREN); return mk_disj(t1, t2)
        if op == OP_IMP:   c.advance(); t2 = _decode_term(hdr, c); c.expect(RPAREN); return mk_imp(t1, t2)
        # application
        acc = t1
        while c.peek() != RPAREN:
            acc = mk_comb(acc, _decode_term(hdr, c))
        c.expect(RPAREN); return acc
    if is_var(tok):
        c.advance(); c.expect(COLON); ty = _decode_type(hdr, c)
        return mk_var(_var_name_in(hdr, tok - VAR_FIRST), ty)
    if tok == OP_EQ:
        c.advance(); return mk_const("=", _const_type("="))
    if is_name(tok):
        c.advance(); name = hdr.names[tok - NAME_FIRST]
        ty = _const_type(name) or _extra_const_types.get(name)
        if ty is None:
            raise RuntimeError(f"decode_term: unknown constant {name}")
        return mk_const(name, ty)
    raise RuntimeError(f"decode_term: unexpected {tok_str(tok)} @ {c.pos}")


def _decode_witness(hdr, c):
    if c.peek() == RPAREN:
        c.advance(); return ("none",)
    c.expect(LPAREN)
    tag = c.peek(); c.advance()
    w = None
    if tag == KW_TERM:
        c.expect(QUOTE); t = _decode_term(hdr, c); c.expect(QUOTE); c.expect(RPAREN)
        w = ("term", t)
    elif tag == KW_TYPE:
        c.expect(QUOTE); ty = _decode_type(hdr, c); c.expect(QUOTE); c.expect(RPAREN)
        w = ("type", ty)
    elif tag == KW_VAR:
        c.expect(QUOTE); vt = c.peek(); c.advance(); c.expect(QUOTE)
        c.expect(QUOTE); ty = _decode_type(hdr, c); c.expect(QUOTE); c.expect(RPAREN)
        w = ("var", _var_name_in(hdr, vt - VAR_FIRST), ty)
    elif tag == KW_AXIOM:
        c.expect(QUOTE); nt = c.peek(); c.advance(); c.expect(QUOTE); c.expect(RPAREN)
        w = ("axiom", hdr.names[nt - NAME_FIRST])
    elif tag == KW_INST:
        pairs = []
        while c.peek() != RPAREN:
            c.expect(LPAREN); c.expect(KW_SUBST)
            c.expect(QUOTE); v = _decode_term(hdr, c); c.expect(QUOTE)
            c.expect(QUOTE); t = _decode_term(hdr, c); c.expect(QUOTE)
            c.expect(RPAREN); pairs.append((v, t))
        c.advance()
        w = ("inst", pairs)
    elif tag == KW_INSTTYPE:
        pairs = []
        while c.peek() != RPAREN:
            c.expect(LPAREN); c.expect(KW_SUBST)
            c.expect(QUOTE); vt = c.peek(); c.advance(); c.expect(QUOTE)
            c.expect(QUOTE); ty = _decode_type(hdr, c); c.expect(QUOTE)
            c.expect(RPAREN); pairs.append((_var_name_in(hdr, vt - VAR_FIRST), ty))
        c.advance()
        w = ("insttype", pairs)
    elif tag == KW_B_AND_W:
        c.expect(LPAREN); c.expect(KW_BOUND)
        c.expect(QUOTE); vt = c.peek(); c.advance(); c.expect(QUOTE)
        c.expect(QUOTE); ty = _decode_type(hdr, c); c.expect(QUOTE); c.expect(RPAREN)
        c.expect(LPAREN); c.expect(KW_WITNESS)
        c.expect(QUOTE); wt = _decode_term(hdr, c); c.expect(QUOTE); c.expect(RPAREN)
        c.expect(RPAREN)
        w = ("bnw", _var_name_in(hdr, vt - VAR_FIRST), ty, wt)
    else:
        raise RuntimeError(f"unknown witness tag: {tok_str(tag)}")
    c.expect(RPAREN)
    return w


def _decode_step(hdr, c):
    c.expect(LPAREN); c.expect(KW_STEP)
    it = c.peek()
    if not is_int_tok(it): raise RuntimeError("expected INT")
    c.advance(); step_id = it - INT_FIRST
    c.expect(LPAREN); c.expect(KW_RULE)
    rt = c.peek()
    if rt not in RULE_FROM_TOK: raise RuntimeError("expected rule name")
    c.advance(); rule = RULE_FROM_TOK[rt]
    c.expect(RPAREN)
    c.expect(LPAREN); c.expect(KW_WITNESS); w = _decode_witness(hdr, c)
    c.expect(LPAREN); c.expect(KW_PREMISES)
    prems = []
    while c.peek() != RPAREN:
        it = c.peek()
        if not is_int_tok(it): raise RuntimeError("expected INT in premises")
        c.advance(); prems.append(it - INT_FIRST)
    c.advance()
    c.expect(RPAREN)
    return Step(step_id, rule, w, prems)


def decode_cert(hdr: PoolHeader, toks: Sequence[int]) -> Cert:
    c = _Cur(toks)
    c.expect(LPAREN); c.expect(KW_CERT)
    steps = []
    while c.peek() == LPAREN:
        if c.pos + 1 < len(c.toks) and c.toks[c.pos + 1] == KW_CONCL:
            break
        steps.append(_decode_step(hdr, c))
    c.expect(LPAREN); c.expect(KW_CONCL)
    c.expect(QUOTE); concl = _decode_term(hdr, c); c.expect(QUOTE)
    c.expect(RPAREN); c.expect(RPAREN)
    return Cert(steps, concl)


# ---------------------------------------------------------------------------
# Grammar automaton.  Symbols on the parsing stack — mirrors grammar.ml.
# ---------------------------------------------------------------------------

# Symbol tags.
Tok, TokIn, TRule, TInt, TVar, TName, TTyHead = range(7)
NTerm, NAfterLPTerm, NAfterFirstTerm, NType, NAfterLPType, NAfterFirstType = range(7, 13)
NCertBody, NStepOrConclAfterLP = 13, 14
NWitnessBody, NWitnessTagAfterLP = 15, 16
NInstPairs, NInstTypePairs, NPremiseList = 17, 18, 19
NStepBody, NStepFieldAfterLP = 20, 21
NDone = 22

# Stack entries:
#  ('Tok', tid)
#  ('TokIn', frozenset)
#  ('TRule',) / ('TInt',) / ('TVar',) / ('TName',) / ('TTyHead',)
#  ('NTerm',) ... etc.

BINDER_OPS = (OP_FORALL, OP_EXISTS, OP_LAMBDA)
BINARY_OPS = (OP_EQ, OP_CONJ, OP_DISJ, OP_IMP)
WITNESS_TAGS = (KW_TERM, KW_TYPE, KW_VAR, KW_AXIOM, KW_INST, KW_INSTTYPE, KW_B_AND_W)
STEP_FIELD_KWS = (KW_RULE, KW_WITNESS, KW_PREMISES)

def initial_state():
    # Stack uses list-back as the top — so reverse-order of consumption.
    return [
        ('NDone',),
        ('Tok', RPAREN),
        ('NCertBody',),
        ('Tok', KW_CERT),
        ('Tok', LPAREN),
    ]


def is_accepting(state):
    return len(state) == 0 or (len(state) == 1 and state[0][0] == 'NDone')


def _any_pool(first, count):
    return tuple(range(first, first + count))


_ANY_RULE      = _any_pool(RULE_FIRST,  len(RULE_NAMES))
_ANY_INT       = _any_pool(INT_FIRST,   INT_COUNT)
_ANY_VAR       = _any_pool(VAR_FIRST,   VAR_COUNT)
_ANY_NAME      = _any_pool(NAME_FIRST,  NAME_COUNT)
_ANY_TYVAR     = _any_pool(TYVAR_FIRST, TYVAR_COUNT)
_ANY_TYCON     = _any_pool(TYCON_FIRST, TYCON_COUNT)
_TY_HEADS_ALL  = (TY_BOOL, TY_IND, TY_FUN, TY_NAT) + _ANY_TYVAR + _ANY_TYCON


def step(state, tok):
    """Consume one token.  Returns the new stack (a fresh list) or None."""
    # Work on a mutable copy
    s = list(state)
    # We may need to recursively re-step (a non-terminal expansion that then
    # consumes the token).  Iterate until either we consume or reject.
    while True:
        if not s: return None
        top = s[-1]
        kind = top[0]

        if kind == 'Tok':
            if top[1] == tok: s.pop(); return s
            return None
        if kind == 'TokIn':
            if tok in top[1]: s.pop(); return s
            return None
        if kind == 'TRule':
            if is_rule(tok): s.pop(); return s
            return None
        if kind == 'TInt':
            if is_int_tok(tok): s.pop(); return s
            return None
        if kind == 'TVar':
            if is_var(tok): s.pop(); return s
            return None
        if kind == 'TName':
            if is_name(tok): s.pop(); return s
            return None
        if kind == 'TTyHead':
            if is_ty_head(tok): s.pop(); return s
            return None

        # Non-terminals — expand based on lookahead
        s.pop()  # remove the non-terminal; we'll push its expansion

        if kind == 'NTerm':
            if tok == LPAREN:
                s.extend([('NAfterLPTerm',), ('Tok', LPAREN)])
            elif is_var(tok):
                s.extend([('NType',), ('Tok', COLON), ('TVar',)])
            elif is_name(tok):
                s.append(('TName',))
            elif tok == OP_EQ:
                s.append(('Tok', OP_EQ))
            else:
                return None
            continue

        if kind == 'NAfterLPTerm':
            if tok in BINDER_OPS:
                # binder pattern
                s.extend([
                    ('Tok', RPAREN), ('NTerm',), ('Tok', DOT),
                    ('NType',), ('Tok', COLON), ('TVar',),
                    ('TokIn', frozenset(BINDER_OPS)),
                ])
            elif tok == OP_NOT:
                s.extend([('Tok', RPAREN), ('NTerm',), ('Tok', OP_NOT)])
            else:
                s.extend([('NAfterFirstTerm',), ('NTerm',)])
            continue

        if kind == 'NAfterFirstTerm':
            if tok in BINARY_OPS:
                s.extend([('Tok', RPAREN), ('NTerm',),
                          ('TokIn', frozenset(BINARY_OPS))])
            else:
                s.extend([('Tok', RPAREN), ('NTerm',)])
            continue

        if kind == 'NType':
            if tok == LPAREN:
                s.extend([('NAfterLPType',), ('Tok', LPAREN)])
            elif is_ty_head(tok):
                s.append(('TTyHead',))
            else:
                return None
            continue

        if kind == 'NAfterLPType':
            s.extend([('NAfterFirstType',), ('NType',)])
            continue

        if kind == 'NAfterFirstType':
            if tok == ARROW:
                s.extend([('Tok', RPAREN), ('NType',), ('Tok', ARROW)])
            elif tok == COMMA:
                s.extend([('NAfterFirstType',), ('NType',), ('Tok', COMMA)])
            elif tok == RPAREN:
                heads = (TY_BOOL, TY_IND, TY_FUN, TY_NAT) + _ANY_TYCON
                s.extend([('TokIn', frozenset(heads)), ('Tok', RPAREN)])
            else:
                return None
            continue

        if kind == 'NCertBody':
            if tok == LPAREN:
                s.extend([('NStepOrConclAfterLP',), ('Tok', LPAREN)])
            else:
                return None
            continue

        if kind == 'NStepOrConclAfterLP':
            if tok == KW_STEP:
                # Enforce strict step body: ( rule R ) ( witness W ) ( premises P* )
                # The kernel decoder requires this exact order and exactly one
                # of each — the loose NStepBody version below let the model
                # emit double-witness / missing-premises certs that the PDA
                # accepted but the kernel rejected.
                s.extend([
                    ('NCertBody',),                  # next cert element after step
                    ('Tok', RPAREN),                 # close step
                    ('NPremiseList',),               # premises body (self-closing)
                    ('Tok', KW_PREMISES),
                    ('Tok', LPAREN),                 # ( for premises clause
                    ('NWitnessBody',),               # witness body (self-closing)
                    ('Tok', KW_WITNESS),
                    ('Tok', LPAREN),                 # ( for witness clause
                    ('Tok', RPAREN),                 # close rule clause
                    ('TRule',),
                    ('Tok', KW_RULE),
                    ('Tok', LPAREN),                 # ( for rule clause
                    ('TInt',),                       # step ID
                    ('Tok', KW_STEP),                # consume KW_step
                ])
            elif tok == KW_CONCL:
                s.extend([
                    ('Tok', RPAREN), ('Tok', QUOTE), ('NTerm',),
                    ('Tok', QUOTE), ('Tok', KW_CONCL),
                ])
            else:
                return None
            continue

        # NStepBody / NStepFieldAfterLP — dead code, replaced by the strict
        # sequence above.  Left as no-ops to defend against any stale state
        # in the unlikely event some external caller pushed them.
        if kind == 'NStepBody' or kind == 'NStepFieldAfterLP':
            return None

        if kind == 'NWitnessBody':
            if tok == RPAREN:
                s.append(('Tok', RPAREN))
            elif tok == LPAREN:
                s.extend([('Tok', RPAREN), ('NWitnessTagAfterLP',), ('Tok', LPAREN)])
            else:
                return None
            continue

        if kind == 'NWitnessTagAfterLP':
            if tok == KW_TERM:
                s.extend([
                    ('Tok', RPAREN), ('Tok', QUOTE), ('NTerm',),
                    ('Tok', QUOTE), ('Tok', KW_TERM),
                ])
            elif tok == KW_TYPE:
                s.extend([
                    ('Tok', RPAREN), ('Tok', QUOTE), ('NType',),
                    ('Tok', QUOTE), ('Tok', KW_TYPE),
                ])
            elif tok == KW_VAR:
                s.extend([
                    ('Tok', RPAREN),
                    ('Tok', QUOTE), ('NType',), ('Tok', QUOTE),
                    ('Tok', QUOTE), ('TVar',), ('Tok', QUOTE),
                    ('Tok', KW_VAR),
                ])
            elif tok == KW_AXIOM:
                s.extend([
                    ('Tok', RPAREN), ('Tok', QUOTE), ('TName',),
                    ('Tok', QUOTE), ('Tok', KW_AXIOM),
                ])
            elif tok == KW_INST:
                s.extend([('NInstPairs',), ('Tok', KW_INST)])
            elif tok == KW_INSTTYPE:
                s.extend([('NInstTypePairs',), ('Tok', KW_INSTTYPE)])
            elif tok == KW_B_AND_W:
                s.extend([
                    ('Tok', RPAREN),
                    ('Tok', RPAREN), ('Tok', QUOTE), ('NTerm',),
                    ('Tok', QUOTE), ('Tok', KW_WITNESS), ('Tok', LPAREN),
                    ('Tok', RPAREN),
                    ('Tok', QUOTE), ('NType',), ('Tok', QUOTE),
                    ('Tok', QUOTE), ('TVar',), ('Tok', QUOTE),
                    ('Tok', KW_BOUND), ('Tok', LPAREN),
                    ('Tok', KW_B_AND_W),
                ])
            else:
                return None
            continue

        if kind == 'NInstPairs':
            if tok == RPAREN:
                return s   # consume the closing rparen ourselves
            elif tok == LPAREN:
                s.extend([
                    ('NInstPairs',), ('Tok', RPAREN),
                    ('Tok', QUOTE), ('NTerm',), ('Tok', QUOTE),
                    ('Tok', QUOTE), ('NTerm',), ('Tok', QUOTE),
                    ('Tok', KW_SUBST), ('Tok', LPAREN),
                ])
            else:
                return None
            continue

        if kind == 'NInstTypePairs':
            if tok == RPAREN:
                return s
            elif tok == LPAREN:
                s.extend([
                    ('NInstTypePairs',), ('Tok', RPAREN),
                    ('Tok', QUOTE), ('NType',), ('Tok', QUOTE),
                    ('Tok', QUOTE), ('TVar',), ('Tok', QUOTE),
                    ('Tok', KW_SUBST), ('Tok', LPAREN),
                ])
            else:
                return None
            continue

        if kind == 'NPremiseList':
            if tok == RPAREN:
                return s
            elif is_int_tok(tok):
                s.extend([('NPremiseList',), ('TInt',)])
            else:
                return None
            continue

        if kind == 'NDone':
            if tok == BOS or tok == EOS:
                s.pop(); return s
            return None

        # Unhandled non-terminal — shouldn't happen
        return None


def valid_next_mask(state):
    """Bool list of length VOCAB_SIZE: which next tokens are accepted."""
    m = [False] * VOCAB_SIZE
    s = list(state)
    while s:
        top = s[-1]
        kind = top[0]
        if kind == 'Tok':
            m[top[1]] = True; return m
        if kind == 'TokIn':
            for t in top[1]: m[t] = True
            return m
        if kind == 'TRule':
            for t in _ANY_RULE: m[t] = True;
            return m
        if kind == 'TInt':
            for t in _ANY_INT: m[t] = True
            return m
        if kind == 'TVar':
            for t in _ANY_VAR: m[t] = True
            return m
        if kind == 'TName':
            for t in _ANY_NAME: m[t] = True
            return m
        if kind == 'TTyHead':
            for t in _TY_HEADS_ALL: m[t] = True
            return m
        if kind == 'NTerm':
            m[LPAREN] = True; m[OP_EQ] = True
            for t in _ANY_VAR: m[t] = True
            for t in _ANY_NAME: m[t] = True
            return m
        if kind == 'NAfterLPTerm':
            for t in BINDER_OPS: m[t] = True
            m[OP_NOT] = True
            # Plus everything that starts a term
            m[LPAREN] = True; m[OP_EQ] = True
            for t in _ANY_VAR: m[t] = True
            for t in _ANY_NAME: m[t] = True
            return m
        if kind == 'NAfterFirstTerm':
            for t in BINARY_OPS: m[t] = True
            m[LPAREN] = True; m[OP_EQ] = True
            for t in _ANY_VAR: m[t] = True
            for t in _ANY_NAME: m[t] = True
            return m
        if kind == 'NType':
            m[LPAREN] = True
            for t in _TY_HEADS_ALL: m[t] = True
            return m
        if kind == 'NAfterLPType':
            m[LPAREN] = True
            for t in _TY_HEADS_ALL: m[t] = True
            return m
        if kind == 'NAfterFirstType':
            m[ARROW] = True; m[COMMA] = True; m[RPAREN] = True
            return m
        if kind == 'NCertBody':         m[LPAREN] = True; return m
        if kind == 'NStepOrConclAfterLP': m[KW_STEP] = True; m[KW_CONCL] = True; return m
        if kind == 'NStepBody':         m[LPAREN] = True; m[RPAREN] = True; return m
        if kind == 'NStepFieldAfterLP':
            for t in STEP_FIELD_KWS: m[t] = True
            return m
        if kind == 'NWitnessBody':      m[LPAREN] = True; m[RPAREN] = True; return m
        if kind == 'NWitnessTagAfterLP':
            for t in WITNESS_TAGS: m[t] = True
            return m
        if kind == 'NInstPairs':        m[LPAREN] = True; m[RPAREN] = True; return m
        if kind == 'NInstTypePairs':    m[LPAREN] = True; m[RPAREN] = True; return m
        if kind == 'NPremiseList':
            m[RPAREN] = True
            for t in _ANY_INT: m[t] = True
            return m
        if kind == 'NDone':
            m[BOS] = True; m[EOS] = True
            return m
        return m
    return m

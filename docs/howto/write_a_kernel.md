# How do I... write a kernel?

**Persona.** You want to **change what is true**. Maybe you want
intuitionistic logic instead of classical, or set theory instead of
HOL, or a new primitive rule that the existing kernel does not
expose. You are modifying the trusted base.

This persona is by far the most dangerous one. Everyone else's bugs
only cause rejections; a kernel bug can cause V to accept a false
theorem. Move slowly. Audit each change. Write tests.

This guide does **not** ask you to design a new logic from scratch.
It walks you through the safer warm-up: **add one new primitive
inference rule** to the existing HOL kernel and have V accept proofs
that use it.

---

## The kernel's shape (5-minute tour)

| File                                  | Role                                              | LOC  |
|---------------------------------------|---------------------------------------------------|------|
| [`kernel/type.ml`](../../kernel/type.ml)    | HOL types and the type-constructor registry      | ~85  |
| [`kernel/term.ml`](../../kernel/term.ml)    | terms, alpha-equivalence, substitution           | ~230 |
| [`kernel/thm.ml`](../../kernel/thm.ml)      | the opaque `Thm.t`                               | ~30  |
| [`kernel/rules.ml`](../../kernel/rules.ml)  | the primitive inference rules                    | ~230 |
| [`kernel/axioms.ml`](../../kernel/axioms.ml)| primitive axioms + theory-axiom manifest         | ~75  |
| [`kernel/cert.ml`](../../kernel/cert.ml)    | certificate parser + per-step rule dispatch      | ~360 |
| [`kernel/verify.ml`](../../kernel/verify.ml)| the verifier V                                   | ~50  |

The trust boundary is the OCaml library `kernel` (defined in
[`kernel/dune`](../../kernel/dune)). Outside that library, the type
`Thm.t` is opaque — no other module can construct a theorem. Inside
the library, only `rules.ml` and `axioms.ml` build `Thm.t` values
via `Thm.mk`. This is the *entire* mechanism that prevents soundness
holes.

If your change does not touch any of the files above, it cannot
affect soundness. If it does, audit it personally.

---

## Worked example: add a new primitive rule, `SYM`

`SYM` says: from $A \vdash s = t$ infer $A \vdash t = s$. It is sound
(equality is symmetric) and easy to derive from `REFL`, `MK_COMB`,
and `EQ_MP`, but those derivations are tedious to write inside
certificates. Promoting `SYM` to a primitive saves about 5 steps per
use without enlarging the trust base meaningfully.

### Step 1 — implement the rule in `rules.ml`

Add at the bottom of [`kernel/rules.ml`](../../kernel/rules.ml):

```ocaml
(* SYM  A ⊢ s = t  ⟹  A ⊢ t = s  *)
let sym th =
  let (s, t) = try Term.dest_eq (Thm.concl th)
               with _ -> err "SYM: conclusion is not an equation" in
  Thm.mk (Thm.hyps th) (Term.mk_eq t s)
```

A few notes:

- We use `Thm.mk` — the kernel-internal constructor. This is the
  only place outside `rules.ml`/`axioms.ml` where it would be
  visible, and that visibility is what makes our change part of the
  trusted base.
- We preserve the hypotheses `A` from the input theorem. Forgetting
  to do this would be a soundness bug (it would let us derive
  unconditional theorems from conditional ones).
- The side condition is checked: if the conclusion is not an
  equation, we raise `Rule_error` and V will reject the certificate
  that called us.

### Step 2 — register the rule in the certificate dispatcher

In [`kernel/cert.ml`](../../kernel/cert.ml), find `apply_step` and
add a case:

```ocaml
  | "SYM", W_none, [a] -> Rules.sym (prem a)
```

This says: a step `(step N (rule SYM) (witness ()) (premises K))`
calls `Rules.sym` on whatever theorem step K produced.

### Step 3 — document the new rule in `docs/LOGIC.md`

Add a row to the table of primitive rules:

```
SYM   From A ⊢ s = t infer A ⊢ t = s
```

Trust-base extensions must be documented. Anyone reading
[`docs/LOGIC.md`](../LOGIC.md) sees exactly what the kernel admits.

### Step 4 — add a unit test in `tests/test_kernel.ml`

Add inside the test block:

```ocaml
  (* SYM *)
  let a = Term.Var ("a", nat) in
  let b = Term.Var ("b", nat) in
  let th_eq = Rules.assume (Term.mk_eq a b) in
  let th_sym = Rules.sym th_eq in
  check_eq "SYM flips equality" (Thm.concl th_sym) (Term.mk_eq b a);
  check_raises "SYM rejects non-equation"
    (fun () -> Rules.sym (Rules.assume (Term.Var ("p", bool_))));
```

### Step 5 — rebuild and run tests

```
make test
```

You should see two new `PASS` lines, and no regressions.

### Step 6 — try a certificate that uses it

```
mkdir -p /tmp/sym
cat > /tmp/sym/sym.kf <<'EOF'
(goal "(y:ind = x:ind)")
EOF
cat > /tmp/sym/sym.cert <<'EOF'
(cert
  (step 1 (rule REFL) (witness (term "x:ind")) (premises))
  (step 2 (rule SYM)  (witness ())             (premises 1))
  (concl "(x:ind = x:ind)"))
EOF
```

Wait — step 2 should yield `x = x` again (because REFL gave us
`x = x` and SYM swaps the sides — still `x = x`). That's fine for a
sanity check that the new rule is wired up. For a less trivial use,
build up a chain via TRANS so the swap matters.

Run:

```
./_build/default/kernel/kernel_main.exe /tmp/sym/sym.kf /tmp/sym/sym.cert
```

If V rejects with "rule SYM: ..." you have a code bug. If V rejects
with "cert's concl ≠ stated goal" your example is too trivial — fix
the .kf goal.

---

## What you must never do

These are the classic ways to introduce a soundness bug:

- **Construct `Thm.t` from raw data outside the kernel library.** If
  someone needs a theorem, they must go through a rule. If you find
  yourself wanting to export `Thm.mk`, stop and reconsider.
- **Forget to preserve hypotheses.** Every rule that takes premise
  theorems must combine their hypothesis sets correctly (usually via
  `Thm.union_hyps`). Dropping hypotheses turns conditional theorems
  into unconditional ones — instant unsoundness.
- **Forget side conditions.** Rules like `ABS` and `GEN` require the
  bound variable to not appear free in the hypotheses. Skipping that
  check lets you "prove" $\forall x.\ x = 0$ from the conditional
  $x = 0 \vdash x = 0$.
- **Modify the verifier loop.** [`kernel/verify.ml`](../../kernel/verify.ml)
  is 50 lines. If you must change it, audit it line by line; do not
  add tactic execution, do not silently widen acceptance, do not
  introduce caches that can be poisoned.

---

## Larger changes: introducing a new logic

Replacing HOL with, say, intuitionistic logic or ZFC is a much
larger undertaking. The skeleton is the same: keep `Thm.t` opaque,
write your inference rules in `rules.ml`, write your axioms in
`axioms.ml`, register them in `cert.ml`'s dispatcher. The hard
parts are:

- Designing a sound primitive presentation of the logic — typically
  10 to 30 rules and a handful of axioms.
- Building enough derived rules (outside the kernel) to make proofs
  tractable.
- Writing extensive tests, including *negative* tests that confirm
  rules reject inputs that violate their side conditions.

The 1956 letter and the Cook–Reckhow framework do not care which
logic you pick — they care only that V is polynomial-time decidable.
Any logic admitting such a V slots into this architecture.

---

## Where to look next

- [`docs/LOGIC.md`](../LOGIC.md) — the existing rules and axioms; a
  good model for documenting your additions.
- [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md) — why the trust
  boundary lives where it does.
- [`kernel/verify.ml`](../../kernel/verify.ml) — the 50 lines you
  most want to leave alone.
- HOL Light's `fusion.ml` and HOL4's `Thm.sml` — the historical
  models this kernel takes after. Worth reading once you are
  comfortable with this one.

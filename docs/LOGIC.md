# Kernel Logic

The kernel implements **classical higher-order logic** in the style of
Church and HOL Light.

## Types

A HOL type is one of:

- a type variable `'a`, `'b`, …
- an application of a registered type constructor: `bool`, `ind`,
  `fun α β`, plus those declared by theory packages (e.g. `nat`).

Built-in type constructors and their arities:

| Constructor | Arity |
|-------------|-------|
| `bool`      | 0     |
| `ind`       | 0     |
| `fun`       | 2     |

## Terms

A term is one of:

- `Var(name, ty)` — a variable
- `Const(name, ty)` — a registered constant at a specific type instance
- `Comb(f, x)` — application
- `Abs(v, body)` — lambda abstraction (v is a `Var`)

Term equality is up to alpha-conversion.

## Primitive constants

Only one primitive constant is needed: equality, polymorphic at
`'a -> 'a -> bool`. All other connectives are defined in terms of it,
following HOL Light's standard encoding.

## Primitive inference rules (10)

A rule maps theorems to theorems; the kernel mints a fresh `Thm.t`
exactly when the side conditions hold.

1. **REFL** ⊢ `t = t`
2. **TRANS** From `A ⊢ s = t` and `B ⊢ t = u` infer `A ∪ B ⊢ s = u`
3. **MK_COMB** From `A ⊢ f = g` and `B ⊢ x = y` (well-typed) infer
   `A ∪ B ⊢ f x = g y`
4. **ABS** From `A ⊢ s = t` (with `v` not free in `A`) infer
   `A ⊢ λv. s = λv. t`
5. **BETA** ⊢ `(λv. t) v = t`
6. **ASSUME** From a `bool`-typed `p` infer `{p} ⊢ p`
7. **EQ_MP** From `A ⊢ p = q` and `B ⊢ p` infer `A ∪ B ⊢ q`
8. **DEDUCT_ANTISYM_RULE** From `A ⊢ p` and `B ⊢ q` infer
   `(A − {q}) ∪ (B − {p}) ⊢ p = q`
9. **INST** instantiate free variables of a theorem with terms
10. **INST_TYPE** instantiate free type variables of a theorem with types

## Primitive axioms (4)

1. **ETA_AX** ⊢ `λx. t x = t` (for `x` not free in `t`)
2. **SELECT_AX** ⊢ `p x ⇒ p (ε p)` (Hilbert's choice)
3. **INFINITY_AX** ⊢ `∃f: ind → ind. ONE_ONE f ∧ ¬ONTO f`
4. **EM_AX** ⊢ `∀p. p ∨ ¬p` (classical excluded middle)

## Defined connectives

| Symbol | Definition (in λ-calculus over `=`)                       |
|--------|-----------------------------------------------------------|
| `T`    | `(λp. p) = (λp. p)`                                       |
| `F`    | `∀p. p`                                                   |
| `∧`    | `λp q. (λf. f p q) = (λf. f T T)`                         |
| `⇒`    | `λp q. (p ∧ q) = p`                                       |
| `∀`    | `λP. P = λx. T`                                           |
| `¬`    | `λp. p ⇒ F`                                               |
| `∨`    | `λp q. ∀r. (p ⇒ r) ⇒ (q ⇒ r) ⇒ r`                         |
| `∃`    | `λP. ∀q. (∀x. P x ⇒ q) ⇒ q`                               |

These definitions are HOL Light's. The point is that the kernel's only
true primitive logical constant is `=`; everything else is derived.

## Theory extensions

A theory package may, via the standard kernel API:

- register new type constructors with declared arity;
- register new constants with declared (possibly polymorphic) types;
- declare axioms by minting `Thm.t` values via the `Axioms.declare`
  hook — these are listed in the kernel manifest and rendered as
  axioms (not theorems) in the output.

Conservative definitional extensions (definitions whose existence is
provable in HOL) are *not* required to use the axiom hook; that work
is deferred to v0.2.

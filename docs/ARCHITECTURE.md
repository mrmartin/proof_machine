# Architecture

```
   ┌───────────────────────────────────────────────┐
   │   Statement Frontend  (untrusted, per-theory) │
   └────────────────────┬──────────────────────────┘
                        │  elaboration (untrusted)
                        ▼
   ┌───────────────────────────────────────────────┐
   │   Kernel Formula  φ : Term.t                  │
   └──────────┬─────────────────────────┬──────────┘
              │                         │
              ▼                         ▼
   ┌──────────────────┐      ┌────────────────────┐
   │ Prover Backend P │      │  Renderer R        │
   │   (untrusted)    │ ─π─► │  cert → LaTeX      │
   │ lookup | enum |  │      │                    │
   │ scripted | …     │      │                    │
   └────────┬─────────┘      └────────────────────┘
            │  certificate π
            ▼
   ┌───────────────────────────────────────────────┐
   │   SEALED KERNEL  V(π, φ) ∈ {accept, reject}   │  ← TRUSTED
   └───────────────────────────────────────────────┘
```

## The trust boundary

The trusted base is `kernel/`:

- `kernel/type.ml`  — HOL types
- `kernel/term.ml`  — HOL terms; alpha-equivalence, substitution
- `kernel/thm.ml`   — the **opaque** `Thm.t` — only the kernel can mint
- `kernel/rules.ml` — the 10 primitive inference rules
- `kernel/axioms.ml`— the 4 primitive axioms + connective definitions
- `kernel/cert.ml`  — certificate format
- `kernel/verify.ml`— V(π, φ)

Everything else is untrusted; bugs there cannot cause V to accept a
non-theorem because every certificate is re-checked against the kernel.

## Soundness argument

Because `Thm.t` is an abstract type whose constructors are private to
the kernel library, no module outside `kernel/` can manufacture a
theorem. The verifier in `verify.ml` reconstructs each step using the
primitive rules in `rules.ml`; if the reconstruction fails or if the
final theorem does not alpha-match φ, V rejects. Thus
**V(π, φ) = accept ⟹ φ is a theorem** in HOL.

## Why this paradigm

See the project's design spec (§O of the project's source paper): the
1956 Gödel question separates the polynomial-time verifier *xBy* from
the search for *x*. The V/P split takes that separation seriously as a
software discipline. Tactics, neural provers, SAT solvers, lookup
caches — anything at all — can produce certificates; only V decides.

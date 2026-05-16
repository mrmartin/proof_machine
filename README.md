# proof_machine

A from-scratch implementation of the **V/P-split** paradigm for proof checking:
a sealed polynomial-time **verifier V** as the sole trust anchor, an
**untrusted prover P** behind a plugin API, **theory packages** giving
mathematicians domain-natural surface syntax, and a **renderer** that turns
machine certificates into readable LaTeX.

Soundness lives in `kernel/` and nowhere else.

## Architecture

```
   surface theorem
        │
        ▼  elaboration (untrusted)
   kernel formula  φ
        │
        ▼  prover backend  (untrusted: lookup, enumerator, scripted, …)
   certificate π
        │
        ▼
   SEALED KERNEL  V(π, φ) ∈ {accept, reject}     ← THE ONLY TRUSTED COMPONENT
```

## Layout

| Directory | Role | Trust |
|---|---|---|
| `kernel/` | HOL kernel, certificate format, verifier V | **trusted** |
| `frontend/` | Surface parser, elaborator, type inference | untrusted |
| `theories/number_theory/` | Notation, constants, declared axioms | declared axioms = explicit trust |
| `provers/` | Plugin prover backends | untrusted |
| `render/` | LaTeX renderer | untrusted |
| `examples/` | Source theorems (tiny + Euclid) | — |
| `tests/` | Unit tests + adversarial certificates + e2e | — |
| `docs/` | Architecture, logic spec, certificate format, prover API | — |

## Build & run

```
make build         # build all OCaml binaries
make demo          # elaborate → prove → verify → render Euclid
make test          # unit tests, adversarial rejection, e2e pipeline
```

## How do I...

Five persona-specific guides, each with a worked example you can replicate
end-to-end:

| If you... | Read this |
|---|---|
| have a `.kf` and a `.cert` and just want V to check it | [Verify a proof](docs/howto/verify_a_proof.md) |
| have an informal theorem and want to write it down in the kernel logic | [Formalize a theorem](docs/howto/formalize_a_theorem.md) |
| have a kernel formula and want to produce a certificate for it | [Prove a theorem](docs/howto/prove_a_theorem.md) |
| want to plug a new prover backend (tactic engine, ML model, SAT solver) into the pipeline | [Write a theorem prover](docs/howto/write_a_theorem_prover.md) |
| want to extend or replace the trusted kernel V itself | [Write a kernel](docs/howto/write_a_kernel.md) |

## Logic

Classical higher-order logic à la Church / HOL Light:
polymorphic simple types, primitive equality, extensionality, choice,
infinity, and classical excluded middle. See `docs/LOGIC.md`.

## Certificates

S-expression-based DAG of inference steps; one step per line for
streaming verification. See `docs/CERTIFICATE.md`.

## Prover plugin API

Any program matching `(φ, ctx, budget, hints) → Stream<Cert>` is a prover.
The kernel filters by re-verifying every emitted certificate.
See `docs/PROVER_API.md`.

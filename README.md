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

## Training a proof-finding GPT

Three trainers live in the repo, each successively more aligned with how
scaled proof-finding GPTs actually work:

| Script | Method | Status |
|---|---|---|
| `microgpt_with_RL_hol.py` | CPU REINFORCE with EMA baseline, scalar-`Value` autograd | Original MVP; 5/7 toy seeds (R5 baseline) |
| `microgpt_with_RL_hol_gpu.py` | GPU REINFORCE (PyTorch, n_layer=2, n_embd=64), batched rollouts + thread-pooled kernel verifiers | ~14× speedup; same 5/7 plateau |
| `microgpt_expit.py` | **Expert iteration** — kernel filters successes, SFT on the buffer of discovered proofs, repeat | Current best |

Each reward is a real `V(π, φ)` from the trusted kernel via
`bin/verify_tokens.exe` — no proxy critic, no estimated value function.

Companion library: a lexeme-level **tokenizer** for HOL theorems &
certs — `tokenizer/` (OCaml, used by the verifier) and
`tokenizer/python/hol_tokenizer.py` (used by the trainer), with a
grammar PDA that gives a sound `valid_next_mask` for constrained
decoding.

See [`docs/RL_TRAINING.md`](docs/RL_TRAINING.md) for the full pipeline
and env-var reference.

### Findings

A series of controlled experiments on the 23-seed test set has pinned
down what these trainers actually do at this scale (≈130K-param
2-layer transformer, 64–128 samples per goal, T ≤ 1.2, 15–150 rounds
of expert iteration).

**1. REINFORCE + EMA baseline is bottlenecked by zero-gradient stuck
arms.**  Failing prompts saturate their baseline at the current
reward, advantage `(R − b)` collapses to zero, gradient becomes zero.
Plateaus at 5/7 toy seeds; the imp seeds never unstick despite
40,000+ rollouts.  See `docs/RL_TRAINING.md`.

**2. Expert iteration solves all in-distribution seeds in seconds.**
With a corpus that contains the gold-cert rule shape for every test
seed, ExitIt hits **12/12 in 3 rounds (~25 s wall clock)**.  The kernel
filter + SFT-on-successes loop has no zero-gradient pathology and
converges fast.

**3. ExitIt does not invent new rule combinations.**  Held out 10
test seeds whose gold-cert rule sequences appear in *no* corpus
pattern.  Result: **0/10 solved**, even at 30 minutes / 149 rounds /
8,448 exploration attempts at temperature 1.2.  Three "novel"
rule-sequence shapes did appear in the discovered set, but all are
wasteful variants of corpus patterns (e.g. REFL-REFL-TRANS with a
dead ASSUME step inserted), not new reasoning.

**4. The bottleneck is structural, not capacity.**  We add one
pattern to the corpus (`beta_inst_pair`: 2-step BETA + INST) and the
test seed `beta_inst_identity` (which uses exactly that rule shape)
goes from "V=+0 forever" to **V=+100 at round 1, 32/32 exploration
success rate immediately**.  No transfer: the other 9 OOD seeds
still don't solve, even ones that benefit from INST in a slightly
different context (e.g. `spec_twice` extends ASSUME-SPEC-DISCH by one
SPEC; `eq_trans_impl` fuses ASSUME-ASSUME-MP-DISCH-DISCH skeleton
with REFL-REFL-TRANS body).

**Conclusion: at this scale, ExitIt is template recall, not rule
composition.**  Adding shape X to the corpus unlocks goals provable
by shape X exactly, and no others.  The model's next-token
distribution at the rule-choice position simply has unseen rules at
essentially zero probability.

Genuine rule invention would require: at minimum, exposure to each
rule in at least one corpus example; for harder generalisation, a
search procedure that explores the rule space outside what
temperature-sampled PDA-masked decoding reaches (beam, MCTS,
kernel-evaluated tree search).

### Built-in novelty instrumentation

`microgpt_expit.py` computes `CORPUS_RULE_SHAPES` at module load and
flags every discovered cert whose rule sequence isn't in that set.
Novel certs go to a separate JSONL (`HOL_EXPIT_NOVEL_PATH`); each
round logs both how many novel shapes appeared and a one-line sample
of each.  End-of-run summary aggregates total novel discoveries by
shape.  Test seeds whose gold-cert shape is OOD are tagged
`[OOD shape]` at startup.

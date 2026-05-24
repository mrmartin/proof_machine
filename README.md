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

### Follow-up experiments (M0 / M1 / M2)

The "kernel-evaluated tree search" and "broader corpus" hooks above
were implemented and run.  Architecture stays fixed at
`n_layer=2, n_embd=64, block_size=320` throughout, so each result
isolates a method effect from a capacity effect.

**M0 — sealed `verify_prefix` subcommand.**  `kernel/verify.ml`
exposes a prefix-mode call that returns the theorem table after
applying `k` cert steps; `bin/verify_tokens.ml` carries a new `P <k>`
protocol prefix.  The readback re-uses the same primitive bindings
as `verify`, so soundness is unchanged.  Tests in
`tests/test_verify_prefix.py` exercise full-prefix, every-incremental
prefix, overshoot, and two adversarial corruptions on all 23 seeds.

**M1 — rule-level kernel-pruned tree search at inference.**
`tree_search_infer.py` implements best-first search at the
rule-choice position with top-k branching, kernel-pruned step
expansion (via `verify_prefix`), and a length-normalised log-prob
priority.  Witness keyword and empty-witness shape are pinned to
match the chosen rule (a syntactic constraint encoded directly from
`Cert.apply_step` in `kernel/cert.ml`).  Six configs (k_outer ∈
{3,5,8}, b_inner ∈ {4,8}, uniform-mix ∈ {0,0.1}, alpha ∈ {0.5,1.0},
budget up to 2000 kernel calls/seed) all converge to:

    OOD solved with ExitIt baseline checkpoint: 0 / 10

This matches the prior finding's prediction.  At every config tree
search reaches at most depth 5 on the easier OOD seeds before the
policy emits witness *content* the kernel rejects.  Two seeds
(`mk_comb_impl`, `conj_assoc`) hit depth-0 across configs: even
ASSUME (which appears as a high-prior rule choice) gets a wrong
witness term.  The bottleneck is the policy's lack of a usable prior
*conditional on goal structure* — not the absence of a rule-space
search.

**M2 — backward synthetic corpus generator.**  `synth/backward_gen.py`
applies random kernel rules forward from random ASSUME/REFL seeds,
emitting `(goal, gold_cert)` pairs whose conclusion is whatever the
random walk produced.  A Python shadow of 17 rules drives the walk
(REFL through INST); every emitted proof is re-verified by the
OCaml kernel, and the first 200 in any batch are kernel-checked
unconditionally.  Variant generation via consistent alpha-renaming
boosts per-shape sample counts.  At 98,818 samples the generator
covers **19,267 unique rule-sequence shapes** (vs 20 in the curated
corpus), 0 kernel rejections after the sanity prefix.

Training a fresh ExitIt checkpoint on `curated + synthetic` (87,800
buffer entries, 2000 bootstrap SFT steps, 8 rounds) and re-running
flat sampling on the 23-seed test set:

| Method                 | Solved (greedy) | Solved (T=1.0, 64 samples) | Novel rule shapes |
|---|---|---|---|
| Baseline ExitIt        | 13/23 | 13/23 (warmup + beta_inst_identity) | 0 |
| Synthetic ExitIt       | 4–6/23 (round-to-round) | 13/23 incl. **`conj_of_eqs` (OOD)** with a novel rule sequence | **76 certs across 59 unique novel shapes** |

The synthetic-trained run produces certificates whose rule sequences
are not in the curated corpus baseline.  Solves on warmup seeds use
*invented* compositions of rules (e.g. proving `REFL` of a variable
via `ASSUME-DISCH-REFL` instead of bare `REFL`) — the model is
clearly composing, not template-recalling.  The single OOD solve
(`conj_of_eqs` via a novel shape) is a thin signal but on the right
side of zero, and the 59 unique novel shapes during training is
where composition mostly shows up.

Tree search on top of the synthetic checkpoint did not multiply
further at the configs we tried (1B-b @ 1000 kernel calls/seed
solves 11/23, slightly under flat sampling) — likely because the
witness *content* problem still binds even when the rule prior is
sharper.  Proof-state exposure (M3) is the natural next move and is
unimplemented here.

**Combined takeaway.**  The 0/10 OOD result under tree search alone
confirms that rule-space exploration with an unconditional-witness
policy doesn't compose.  The shift to a synthetic corpus moves the
needle: the model produces 76 verified certs with novel rule
sequences and solves one OOD seed by an invented composition.  That
is the first crack in template-recall at this scale.  The remaining
binding constraint is witness-content selection, which is exactly
what state-conditional decoding (M3) is supposed to fix.

Run artefacts under `runs/`: `m1_eval.log`, `m2_train_v3.log`,
`m2_eval.log`, `m2_eval.csv` (partial), and the synthetic corpus
itself at `hol_synth_corpus_v3.jsonl`.

### Long continuous run (rounds 1–500, adaptive sampler at 201–500)

The first 500-round continuous run produced two phases:

- **Rounds 1–200** (fixed `RULE_BIAS` synth sampler).  Buffer grew to
  483K entries.  Greedy solve rate plateaued at a mean of **9.77/23**
  with the OOD seeds wobbling between 0 and 1 solves and several
  in-distribution seeds (`conj_swap`, `k_combinator`) under 20%.
- **Rounds 201–500** (adaptive inverse-frequency synth sampler at
  `HOL_EXPIT_SYNTH_TEMP=2.0`).  Buffer grew to 1.13M entries.  Greedy
  solve rate rose to a mean of **13.34/23** (rolling mean of the last
  100 rounds: **14.09/23**) and three previously-uncrackable OOD
  seeds began solving.

The change between phases is one and only one thing: the synth
generator's rule-choice distribution.  Architecture, kernel, PDA,
SFT loss, exploration temperature, and verifier are all unchanged.
See [the adaptive-sampler section in CLAUDE.md](CLAUDE.md) for the
exact scoring formula and tuning history.

**Per-seed solve rate, phase comparison.**

| Seed | Phase 1 (≤200) | Phase 2 (201–500) | Δ | Note |
|---|---:|---:|---:|---|
| `refl_x`/`y`/`n`                 |  99% |100% | +1  | warmup |
| `refl_p_bool`/`q_bool`           |  96% |100% | +4  | warmup |
| `imp_p` / `imp_q`                |  70% | 98% | +28 | the originally REINFORCE-stuck seeds |
| `prime_imp_prime`                |  59% | 99% | +40 |     |
| `conj_proj_left`                 |  71% | 98% | +28 |     |
| `conj_swap`                      |  14% | 52% | +38 |     |
| `k_combinator`                   |  17% | 63% | +46 |     |
| `gcd_refl`                       |  98% | 95% | −4  | mild regression |
| `disj_imp_self`                  |  59% | 98% | +39 |     |
| `conj_of_eqs`                    |  19% | 28% | +9  | OOD |
| `abs_impl`                       |   0% | 16% | +16 | **first solve, OOD** |
| `spec_twice`                     |   0% |  1% | +1  | **first solve, OOD** (`ASSUME→SPEC→SPEC→DISCH`) |
| `eq_trans_impl`                  |   0% |  1% | +1  | **first solve, OOD** |
| `comp_imp`                       |   0% |  1% | +1  | **first solve, OOD** |
| `beta_inst_identity`             |   0% | 83% |+83  | **first solve, OOD** (was the structural-bottleneck control) |
| `double_gen_imp`                 |   2% |  1% | −1  | unchanged |
| `mk_comb_impl`                   |   0% |  0% |  0  | still stuck |
| `conj_assoc`                     |   0% |  0% |  0  | still stuck |
| `triple_conj_intro`              |   0% |  0% |  0  | still stuck |

Peak round: **17/23 solved** in two distinct rounds (e.g. at round
~464, where `spec_twice` solved alongside everything in the
"reliably-solved" cluster).  The 17/23 ceiling is structural —
`mk_comb_impl`, `conj_assoc`, `triple_conj_intro` and
`double_gen_imp` use rule shapes the adaptive sampler hasn't yet
made common enough at their required positions.

**Cumulative novelty.**  From the end-of-run novelty summary:

- corpus baseline rule-sequence shapes: **20**
- novel certs discovered:               **64,141**
- across unique novel shapes:           **28,009**

That is a ≈1,400× expansion of the rule-shape distribution the
model has been trained on, all kernel-verified.  Most of those
shapes are not solving test seeds, but the existence of the
`beta_inst_identity` / `abs_impl` / `spec_twice` solves shows the
SFT loss is in fact transferring compositional structure out of the
synthetic distribution into the policy.

**The `beta_inst_identity` unlock is the cleanest signal.**  This
seed was previously the canonical "structural bottleneck"
demonstration: in the M2 README paragraph above, adding its rule
shape to the corpus took it from V=+0 to 32/32 success
instantaneously, while no other OOD seed transferred.  Under the
adaptive sampler, the seed solves greedily in 83% of evaluation
rounds *without* any hand-curated insertion of its shape — the
generator's broader rule distribution made the composition
discoverable on its own.

**Throughput.**  Each round of phase 2 runs in ~75 s on the
trainer's single GPU: ~7 s exploration, ~40 s synth (of which ~32 s
is the per-round buffer scan to rebuild the position-frequency
table), ~25 s SFT, ~14 s eval.  Buffer scan cost grows linearly
with buffer size; at 1.13M entries it has not yet become the
bottleneck but will overtake SFT around 2–3M entries.  A persistent
incrementally-updated counter would erase that cost when needed.

**What is *not* claimed.**  The 17/23 peak is one round, not a
trend; the rolling-100-round mean is 14.09.  Greedy eval is also a
single sample — temperature sampling at T=1.0 with 64 attempts per
seed (what the M2 paragraph reported) would solve more.  And the
four still-unsolved seeds need either Change 2/3 of the sampler
work, more synthetic diversity at the right positions, or M3-style
proof-state exposure to fix.  The combined takeaway from the M2
section above — *witness-content selection is the binding
constraint* — still holds: most of the OOD-seed misses we see at
greedy eval are now correct-rule-sequence-wrong-witness, not
correct-witness-wrong-rule.

### Built-in novelty instrumentation

`microgpt_expit.py` computes `CORPUS_RULE_SHAPES` at module load and
flags every discovered cert whose rule sequence isn't in that set.
Novel certs go to a separate JSONL (`HOL_EXPIT_NOVEL_PATH`); each
round logs both how many novel shapes appeared and a one-line sample
of each.  End-of-run summary aggregates total novel discoveries by
shape.  Test seeds whose gold-cert shape is OOD are tagged
`[OOD shape]` at startup.

### Resumable continuous training

Training is designed to run indefinitely and survive machine reboots.
The trainer maintains three durable artefacts on disk:

| File                       | Role | Write discipline |
|---|---|---|
| `hol_expit_ckpt.pt`        | model + optimiser state | atomic tmp+rename |
| `hol_expit_buffer.jsonl`   | every accepted (goal_toks, cert_toks) | append + fsync |
| `hol_expit_novel.jsonl`    | verified certs whose rule shape is outside the curated baseline | append + fsync |

On startup `microgpt_expit.py` loads the checkpoint if present
(resuming at `saved_round + 1`) and loads the buffer JSONL verbatim
if present (so accumulated discoveries persist).  Each round writes
both files before declaring the round complete; at most one
in-flight round's outputs are lost on a hard kill.

**Continuous corpus expansion.**  Setting
`HOL_EXPIT_SYNTH_PER_ROUND=K` (default 2000) appends `K` fresh
synthetic-backward samples to the buffer each round.  The samples
are generated by `synth/backward_gen.py` (random forward rule
application, kernel-shadowed; the OCaml kernel re-verifies the first
200 of every batch as a sanity prefix).  The buffer grows
unboundedly; the model sees an ever-wider distribution of rule
compositions in subsequent SFT epochs.

**Launching.**

```bash
make train                     # picks up where the last run left off
# or:
HOL_EXPIT_NUM_ROUNDS=2000 \
HOL_EXPIT_SYNTH_PER_ROUND=2000 \
  bash scripts/train_continuous.sh
```

`scripts/train_continuous.sh` auto-restarts on crash with exponential
backoff (cap 60 s).  Inspect with:

```bash
grep -E "round [0-9]+ total|eval: solved|synthesised|NOVEL" hol_expit.log | tail -50
python3 -c "import torch; print(torch.load('hol_expit_ckpt.pt', weights_only=False)['round'])"
wc -l hol_expit_buffer.jsonl
```

**Re-evaluating without disturbing training.**

```bash
make eval-ood
```

forks its own verifier subprocesses against the current
checkpoint and writes `runs/eval_ood.csv`; safe to run in parallel
with `make train`.

**Hard reset** (rarely needed):

```bash
make reset-train               # deletes ckpt + buffer + novel jsonl
```

See `CLAUDE.md` for the per-knob reference and operational notes
intended for future Claude sessions.

# CLAUDE.md

Operational notes for Claude Code instances working on this repo across
sessions. The conversation history doesn't persist; the on-disk state
does.  Read this file first when you arrive.

## What the project is doing right now

A 130K-parameter, 2-layer transformer is being trained on a
continuously-expanding corpus of kernel-valid HOL Light proofs.  The
goal is not raw performance — it's to demonstrate **rule composition**:
the model should produce certificates whose rule sequences were never
shown to it.  Background and prior findings are in `README.md`'s
"Findings" and "Follow-up experiments" sections.

You should default to **continuing** the existing run rather than
starting over.  The state is durable on disk:

| File                       | Role |
|----------------------------|------|
| `hol_expit_ckpt.pt`        | model + optimiser state; atomic-write |
| `hol_expit_buffer.jsonl`   | every (goal_toks, cert_toks) the trainer has accepted; fsynced after each append |
| `hol_expit_novel.jsonl`    | subset of the buffer whose rule shape is outside the curated baseline |
| `hol_expit.log`            | per-round trainer log |
| `runs/continuous_train.log`| stdout/stderr from `scripts/train_continuous.sh` |

If any of those exist, the run is mid-flight.  Inspect them before
deciding what to do.

## Continuing the training run

```bash
make train         # = bash scripts/train_continuous.sh
```

That:

1. Builds the OCaml verifier if needed (cheap if already built).
2. Loads `hol_expit_ckpt.pt` if it exists — resumes at
   `saved_round + 1`.  Otherwise bootstraps from the curated corpus.
3. Loads `hol_expit_buffer.jsonl` verbatim if it exists; otherwise
   seeds the buffer from `ENCODED_CORPUS` in `microgpt_with_RL_hol.py`.
4. Runs up to `HOL_EXPIT_NUM_ROUNDS` rounds (default 500).  Each round:
   - Exploration: PDA-masked sampling, kernel-verified.
   - Buffer append: every verified cert.
   - **Synthetic corpus expansion**: `HOL_EXPIT_SYNTH_PER_ROUND`
     fresh samples from `synth/backward_gen.py` get appended to the
     buffer.  Default 2000 per round.
   - SFT epoch with this round's discoveries upweighted.
   - Greedy eval against the 23-seed test set.
   - Atomic checkpoint save.
5. Restarts on crash with exponential backoff (cap 60 s).

**Crash safety contract.**  At most one round's exploration outputs
can be lost on power-loss; checkpoint and buffer writes are fsynced.
The checkpoint write uses tmp+rename so a truncated `.pt` is
impossible.

## Tuning the run

All knobs are env vars; defaults are in `scripts/train_continuous.sh`.

| Env var                          | What it does | Default |
|---|---|---|
| `HOL_EXPIT_NUM_ROUNDS`           | total rounds (inclusive of any already done) | 500 |
| `HOL_EXPIT_SAMPLES_PER_GOAL`     | exploration samples per seed per round | 32 |
| `HOL_EXPIT_SFT_STEPS`            | SFT steps per round | 400 |
| `HOL_EXPIT_SFT_BATCH`            | SFT batch size | 128 |
| `HOL_EXPIT_TEMPERATURE`          | exploration temperature | 1.0 |
| `HOL_EXPIT_UPWEIGHT`             | weight on this-round discoveries vs old | 4.0 |
| `HOL_EXPIT_LR`                   | AdamW lr | 3e-4 |
| `HOL_EXPIT_SYNTH_PER_ROUND`      | synth samples appended per round | 2000 |
| `HOL_EXPIT_SYNTH_MIN_DEPTH`      | min synth proof depth | 2 |
| `HOL_EXPIT_SYNTH_MAX_DEPTH`      | max synth proof depth | 10 |
| `HOL_EXPIT_SYNTH_TEMP`           | adaptive inverse-freq sampler temperature (lower = stronger rare-rule boost; T=1.0 over-corrects to BETA at position 0; T=2.0 is the empirical sweet spot) | 2.0 |

When you change one of these, the change applies to **future rounds
only** — the corpus + checkpoint from prior rounds are kept.

### Adaptive rule sampler (synth/backward_gen.py)

The synthetic generator no longer uses a fixed `RULE_BIAS`.  Each
round, `expand_corpus_synthetic` scans `hol_expit_buffer.jsonl` once
(~15 s on a 500K-entry buffer) to build a position-aware rule
frequency table, then samples each rule choice by

    score(r) = RULE_BIAS[r] * (succ_rate[r] + ε) / (freq[pos][r] + ε)

with `succ_rate` tracked online for the round, softmax over log-scores
at `HOL_EXPIT_SYNTH_TEMP`.  The within-round `freq.bump` causes the
batch to self-balance.

This fixes the corpus position-0 entropy collapse (75% ASSUME / 25%
REFL) but introduces a known asymmetry: BETA is rare in the corpus
*and* easy to apply, so at low temperature it over-corrects.  T=2.0
is a tuned trade-off.  If you see BETA dominating position 0 of new
samples, raise T to 3–4.  If TRANS / MK_COMB are still under 1%, lower
T to 1.5.

## Inspecting progress

```bash
# Round-by-round summary
grep -E "round [0-9]+ total|eval: solved|exploration:|synthesised|NOVEL" hol_expit.log | tail -50

# Current checkpoint round
python3 -c "import torch; c=torch.load('hol_expit_ckpt.pt', weights_only=False); print(c['round'])"

# Buffer size
wc -l hol_expit_buffer.jsonl

# Unique rule shapes the model has seen
python3 -c "
import json, sys
shapes = set()
sys.path.insert(0, 'tokenizer/python')
import hol_tokenizer as T
import microgpt_expit as E
with open('hol_expit_buffer.jsonl') as f:
    for line in f:
        d = json.loads(line)
        shapes.add(E.cert_rule_shape(d['cert_toks']))
print(f'{len(shapes)} unique rule shapes in buffer')
"
```

## Hard reset

```bash
make reset-train   # deletes ckpt + buffer + novel
```

Use sparingly.  The default behaviour is to continue forever; resets
discard accumulated discoveries.  If you do reset, the next run takes
the curated corpus as the starting buffer.

## Evaluating without disturbing the run

The training process holds nothing the eval harness needs.  You can
evaluate the **current** checkpoint at any time:

```bash
make eval-ood
# or directly:
HOL_EXPIT_CKPT_PATH=$PWD/hol_expit_ckpt.pt python3 eval_ood.py \
  --ckpt $PWD/hol_expit_ckpt.pt --methods 1A-T1.0 1B-b \
  --out runs/eval_$(date +%s).csv
```

The eval forks its own verifier subprocesses; it doesn't read the
buffer or touch the checkpoint file.  Running concurrently with
training is safe.

## When you should start fresh

Almost never.  If you're tempted, ask yourself:

- Is the architecture changing?  (n_layer, n_embd, block_size, vocab.)
  Then yes — the old checkpoint won't load.  Use `make reset-train`.
- Has the rule alphabet or PDA grammar changed?  Then yes — token IDs
  shift and old buffer entries are invalid.  Reset.
- Are you debugging a generator bug?  No — the kernel-reverify path
  ensures only valid samples enter the buffer.  Trust the existing
  data.
- Did you change SFT hyperparameters?  No — the new round uses the
  new settings against the existing checkpoint + buffer.

## When the milestone is M3 or beyond

`README.md` lists the planned milestones (`M3 = proof-state exposure
in prompt`).  When M3 lands it will:

- Extend the PDA grammar to accept `(state ...)` blocks
  interleaved with `(step ...)`.
- Change the corpus format so each step is preceded by its derived
  theorem table.
- That's a *grammar change* — token IDs may shift.  Hard-reset the
  buffer and checkpoint before training the M3 corpus.

## Don't ever

- Hand-edit `hol_expit_buffer.jsonl`.  It's an append-only log; any
  in-place edit corrupts the dedup keys set on the next load.
- Resume training across a kernel/verifier change without `make build`
  first.  The trainer caches the verifier subprocess; if the binary
  changed but the cache hits a stale path, results may silently
  diverge from re-verified ones.
- Commit `hol_expit_*.{pt,jsonl}` or `runs/`.  They're in `.gitignore`
  and they're large.  The accumulated state is the user's; git is
  for code.

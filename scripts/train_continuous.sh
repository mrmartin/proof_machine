#!/usr/bin/env bash
# scripts/train_continuous.sh — long-running ExitIt trainer that
# resumes from a checkpoint after a machine reboot or kill, and
# continuously expands the corpus with synthetic-backward samples.
#
# Usage:
#   scripts/train_continuous.sh                  # default 500 rounds
#   HOL_EXPIT_NUM_ROUNDS=2000 scripts/train_continuous.sh
#
# State files (all under repo root, all written atomically):
#
#   hol_expit_ckpt.pt           — model + optimiser; written each round
#                                  via tmp+rename so a crash never leaves
#                                  a corrupt checkpoint
#   hol_expit_buffer.jsonl      — (goal_toks, cert_toks) buffer; appended
#                                  with fsync per round
#   hol_expit_novel.jsonl       — verified certs with rule-sequence
#                                  shapes outside the curated baseline
#   hol_expit.log               — round-by-round log
#
# Resume contract: if hol_expit_ckpt.pt exists at startup, the trainer
# loads it and continues from `round + 1`.  If the buffer JSONL exists,
# it's loaded verbatim.  Both files are durable on disk; deleting
# either of them is the *only* way to reset the run.

set -u
cd "$(dirname "$0")/.."

# Build the verifier if missing.  Cheap to re-run when already built.
make build >/dev/null 2>&1

# Defaults are tuned for "leave running indefinitely" — small rounds,
# many of them, with continuous corpus expansion.
export HOL_EXPIT_NUM_ROUNDS="${HOL_EXPIT_NUM_ROUNDS:-500}"
export HOL_EXPIT_SAMPLES_PER_GOAL="${HOL_EXPIT_SAMPLES_PER_GOAL:-32}"
export HOL_EXPIT_SFT_STEPS="${HOL_EXPIT_SFT_STEPS:-400}"
export HOL_EXPIT_SFT_BATCH="${HOL_EXPIT_SFT_BATCH:-128}"
export HOL_EXPIT_TEMPERATURE="${HOL_EXPIT_TEMPERATURE:-1.0}"
export HOL_EXPIT_LR="${HOL_EXPIT_LR:-3e-4}"
export HOL_EXPIT_UPWEIGHT="${HOL_EXPIT_UPWEIGHT:-4.0}"
# Continuous corpus expansion: append this many fresh synthetic-backward
# samples to the buffer at the end of every round.  Set to 0 to disable.
export HOL_EXPIT_SYNTH_PER_ROUND="${HOL_EXPIT_SYNTH_PER_ROUND:-2000}"
export HOL_EXPIT_SYNTH_MIN_DEPTH="${HOL_EXPIT_SYNTH_MIN_DEPTH:-2}"
export HOL_EXPIT_SYNTH_MAX_DEPTH="${HOL_EXPIT_SYNTH_MAX_DEPTH:-10}"
# Change 2: probability that a walk uses a structured premise-kit
# prefix instead of the random ASSUME/REFL/BETA seed loop.  Set to
# 0.0 to disable (exact backward-compat).  See synth/backward_gen.py
# KITS for the kit registry.
export HOL_SYNTH_KIT_PROB="${HOL_SYNTH_KIT_PROB:-0.5}"

# Auto-restart on crash with exponential backoff capped at 60s.
sleep_s=1
while true; do
  python3 -u microgpt_expit.py
  status=$?
  if [ $status -eq 0 ]; then
    echo "[$(date)] trainer exited cleanly (NUM_ROUNDS reached)"
    break
  fi
  echo "[$(date)] trainer exited with status $status, restarting in ${sleep_s}s"
  sleep "$sleep_s"
  if [ "$sleep_s" -lt 60 ]; then
    sleep_s=$((sleep_s * 2))
  fi
done

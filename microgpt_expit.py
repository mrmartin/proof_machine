"""microgpt_expit.py — expert-iteration trainer for HOL proof-finding GPT.

Replaces REINFORCE with the architecture every scaled proof-finding GPT
(GPT-f, DeepSeek-Prover-V1/V2, Goedel-Prover, HTPS) actually uses:

    1. Propose: sample N candidate certs per goal at temperature T.
    2. Filter:  kernel-verify; keep only +100s.
    3. Buffer:  append (goal, cert) successes to a persistent JSONL.
    4. SFT:     train one epoch over the buffer, upweighting this round's
                discoveries.
    5. Repeat.

The kernel filters successes; failures are discarded entirely. No advantage
estimation, no baseline, no policy gradient. Exploration only has to succeed
at *discovery*, not at *learning*.

The model class, VerifierPool, and seed/corpus data are imported from the
existing GPU trainer.

Run:

    HOL_EXPIT_NUM_ROUNDS=5 HOL_EXPIT_SAMPLES_PER_GOAL=32 \\
      python3 microgpt_expit.py
"""

import os
import sys
import json
import time
import random
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tokenizer", "python"))
sys.path.insert(0, HERE)
import hol_tokenizer as T  # noqa: E402

# Reuse model + verifier pool from the GPU REINFORCE trainer.
from microgpt_with_RL_hol_gpu import (  # noqa: E402
    HOLGPT, VerifierPool,
    N_LAYER, N_HEAD, N_EMBD, HEAD_DIM,
)
# Reuse seed + corpus data from the CPU trainer.
from microgpt_with_RL_hol import (  # noqa: E402
    SEEDS, encoded_goals, ENCODED_CORPUS,
)


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------
LOG_PATH         = os.environ.get("HOL_EXPIT_LOG_PATH",    os.path.join(HERE, "hol_expit.log"))
BUFFER_PATH      = os.environ.get("HOL_EXPIT_BUFFER_PATH", os.path.join(HERE, "hol_expit_buffer.jsonl"))
CKPT_PATH        = os.environ.get("HOL_EXPIT_CKPT_PATH",   os.path.join(HERE, "hol_expit_ckpt.pt"))
NUM_ROUNDS       = int(os.environ.get("HOL_EXPIT_NUM_ROUNDS",       "10"))
SAMPLES_PER_GOAL = int(os.environ.get("HOL_EXPIT_SAMPLES_PER_GOAL", "32"))
TEMPERATURE      = float(os.environ.get("HOL_EXPIT_TEMPERATURE",    "1.0"))
SFT_BATCH        = int(os.environ.get("HOL_EXPIT_SFT_BATCH",        "64"))
SFT_STEPS        = int(os.environ.get("HOL_EXPIT_SFT_STEPS",        "200"))
UPWEIGHT         = float(os.environ.get("HOL_EXPIT_UPWEIGHT",       "4.0"))
LR               = float(os.environ.get("HOL_EXPIT_LR",             "3e-4"))
BLOCK_SIZE       = int(os.environ.get("HOL_EXPIT_BLOCK_SIZE",       "192"))
MAX_GEN          = int(os.environ.get("HOL_EXPIT_MAX_GEN",          "160"))
VERIFIER_THREADS = int(os.environ.get("HOL_EXPIT_VERIFIER_THREADS",
                                       str(max(1, (os.cpu_count() or 4) - 4))))
SEED             = int(os.environ.get("HOL_EXPIT_SEED", "42"))
VERIFIER_BIN     = os.environ.get(
    "HOL_VERIFIER_BIN",
    os.path.join(HERE, "_build", "default", "bin", "verify_tokens.exe"),
)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Logging.
# ---------------------------------------------------------------------------
_log_f = open(LOG_PATH, "a", buffering=1)
def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    _log_f.write(line + "\n")


# ---------------------------------------------------------------------------
# Proof buffer.
# ---------------------------------------------------------------------------
@dataclass
class BufferEntry:
    goal_toks: List[int]
    cert_toks: List[int]
    source: str                # "corpus" | "discovered"
    discovered_round: int      # 0 for corpus seeds; N for round-N discoveries


def _key(e: BufferEntry) -> Tuple[tuple, tuple]:
    return (tuple(e.goal_toks), tuple(e.cert_toks))


def load_buffer_jsonl(path: str) -> Tuple[List[BufferEntry], Set[Tuple[tuple, tuple]]]:
    """Load all entries from JSONL.  Keys set tracks the unique
    (goal_toks, cert_toks) pairs for *future* discovery dedup; we don't dedup
    the entries themselves — duplicates affect the SFT sampling distribution
    (e.g. the 1600 corpus replicas collapse to 8 unique sequences under
    canonical tokenization but the replication is what gives SFT enough
    SGD step count)."""
    entries: List[BufferEntry] = []
    keys: Set[Tuple[tuple, tuple]] = set()
    if not os.path.exists(path):
        return entries, keys
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            e = BufferEntry(
                goal_toks=list(d["goal_toks"]),
                cert_toks=list(d["cert_toks"]),
                source=d.get("source", "corpus"),
                discovered_round=int(d.get("discovered_round", 0)),
            )
            entries.append(e)
            keys.add(_key(e))
    return entries, keys


def append_buffer_jsonl(path: str, entries: List[BufferEntry]):
    with open(path, "a") as f:
        for e in entries:
            f.write(json.dumps(asdict(e)) + "\n")


def init_buffer_from_corpus() -> Tuple[List[BufferEntry], Set[Tuple[tuple, tuple]]]:
    """Seed the buffer with all 1600 corpus entries (do NOT dedup — the
    replicas are deliberately what gives SFT enough gradient updates).
    The keys set still records the unique sequences for discovery dedup."""
    entries: List[BufferEntry] = []
    keys: Set[Tuple[tuple, tuple]] = set()
    for goal_toks, cert_toks in ENCODED_CORPUS:
        e = BufferEntry(
            goal_toks=list(goal_toks),
            cert_toks=list(cert_toks),
            source="corpus",
            discovered_round=0,
        )
        entries.append(e)
        keys.add(_key(e))
    return entries, keys


# ---------------------------------------------------------------------------
# Sampling (no-grad, temperature, PDA-masked).
# ---------------------------------------------------------------------------
def sample_rollouts(model: HOLGPT,
                    prompts: List[List[int]],
                    max_gen: int,
                    block_size: int,
                    temperature: float,
                    device: torch.device,
                    greedy: bool = False) -> List[List[int]]:
    """No-grad PDA-masked sampling.  Returns the list of generated token
    sequences (each rollout's emitted tokens, not including the prompt).
    `greedy=True` overrides `temperature` and takes argmax.

    Groups rollouts by prompt length and processes each group separately —
    if we batched ragged prompts, PAD tokens would sit between each prompt
    and its first sampled token, breaking position-embedding alignment vs
    training (where the cert always starts immediately after the prompt)."""
    if not prompts:
        return []
    by_len: dict = {}
    for i, p in enumerate(prompts):
        by_len.setdefault(len(p), []).append(i)
    results: List[Optional[List[int]]] = [None] * len(prompts)
    for L_, idxs in by_len.items():
        sub_prompts = [prompts[i] for i in idxs]
        sub_gens = _sample_uniform_length(model, sub_prompts, max_gen,
                                          block_size, temperature, device, greedy)
        for i, g in zip(idxs, sub_gens):
            results[i] = g
    return [g if g is not None else [] for g in results]


def _sample_uniform_length(model: HOLGPT,
                           prompts: List[List[int]],
                           max_gen: int,
                           block_size: int,
                           temperature: float,
                           device: torch.device,
                           greedy: bool) -> List[List[int]]:
    """Inner sampler: all prompts must have the same length, so no padding
    is needed between prompt and sampled tokens."""
    B = len(prompts)
    if B == 0:
        return []
    L = len(prompts[0])
    cur = torch.tensor(prompts, dtype=torch.long, device=device)        # [B, L]

    grammar = [T.initial_state() for _ in range(B)]
    live = [True] * B
    pos_next = [L] * B
    gen_toks_h: List[List[int]] = [[] for _ in range(B)]

    arange_B = torch.arange(B, device=device)

    with torch.no_grad():
        for _t in range(max_gen):
            if cur.shape[1] >= block_size:
                break
            if not any(live):
                break

            logits_all = model(cur)                         # [B, T_cur, V]
            idx_last = torch.tensor(
                [max(0, pn - 1) for pn in pos_next],
                dtype=torch.long, device=device,
            )
            logits = logits_all[arange_B, idx_last]         # [B, V]

            # Per-sequence grammar mask on CPU.  Dead rollouts get a PAD-only
            # mask so multinomial doesn't NaN; we never read their output.
            mask_rows = []
            for b in range(B):
                if live[b]:
                    mask_rows.append(T.valid_next_mask(grammar[b]))
                else:
                    row = [False] * T.VOCAB_SIZE
                    row[T.PAD] = True
                    mask_rows.append(row)
            mask = torch.tensor(mask_rows, dtype=torch.bool, device=device)

            masked = logits.masked_fill(~mask, float("-inf"))
            if greedy:
                # argmax over allowed tokens.
                sampled = masked.argmax(dim=-1)
            else:
                probs = F.softmax(masked / max(temperature, 1e-6), dim=-1)
                row_ok = mask.any(dim=1)
                if not bool(row_ok.all().item()):
                    safe = probs.clone()
                    safe[~row_ok] = 0.0
                    safe[~row_ok, T.PAD] = 1.0
                    sampled = torch.multinomial(safe, num_samples=1).squeeze(-1)
                else:
                    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

            sampled_h = sampled.tolist()
            for b in range(B):
                if not live[b]:
                    continue
                tk = sampled_h[b]
                ns = T.step(grammar[b], tk)
                if ns is None:
                    live[b] = False
                    continue
                grammar[b] = ns
                gen_toks_h[b].append(tk)
                if T.is_accepting(grammar[b]) or len(gen_toks_h[b]) >= max_gen:
                    live[b] = False

            cur = torch.cat([cur, sampled.unsqueeze(1)], dim=1)
            for b in range(B):
                pos_next[b] = cur.shape[1]

    return gen_toks_h


# ---------------------------------------------------------------------------
# SFT epoch.
# ---------------------------------------------------------------------------
def run_sft_epoch(model: HOLGPT, optim, buffer: List[BufferEntry],
                  batch: int, weights: List[float],
                  n_steps: int) -> float:
    """Run n_steps SFT updates with weighted resampling.  Returns mean loss."""
    model.train()
    total_loss = 0.0
    for _step in range(n_steps):
        picks = random.choices(buffer, weights=weights, k=batch)
        full_seqs = []
        prompt_lens = []
        for e in picks:
            prompt = [T.BOS] + list(e.goal_toks) + [T.BOS]
            target = list(e.cert_toks) + [T.EOS]
            full_seqs.append(prompt + target)
            prompt_lens.append(len(prompt))
        T_full = min(BLOCK_SIZE, max(len(f) for f in full_seqs))
        x = torch.full((batch, T_full), T.PAD, dtype=torch.long, device=DEVICE)
        for b, f in enumerate(full_seqs):
            ff = f[:T_full]
            x[b, :len(ff)] = torch.tensor(ff, dtype=torch.long, device=DEVICE)
        logits = model(x[:, :-1])
        targets = x[:, 1:].clone()
        pos = torch.arange(T_full - 1, device=DEVICE).unsqueeze(0)
        plen = torch.tensor(prompt_lens, device=DEVICE).unsqueeze(1)
        ignore = pos < (plen - 1)
        targets[ignore] = T.PAD
        loss = F.cross_entropy(
            logits.reshape(-1, T.VOCAB_SIZE),
            targets.reshape(-1),
            ignore_index=T.PAD,
        )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total_loss += loss.item()
    return total_loss / max(1, n_steps)


# ---------------------------------------------------------------------------
# Greedy eval.
# ---------------------------------------------------------------------------
def greedy_eval(model: HOLGPT, vpool: VerifierPool) -> List[Tuple[str, int, int]]:
    """For each test seed, greedy decode and kernel-verify.  Returns list of
    (label, gen_len, verdict)."""
    model.eval()
    prompts = [[T.BOS] + list(gt) + [T.BOS] for (_, _, gt, _) in encoded_goals]
    proposals = sample_rollouts(model, prompts, MAX_GEN, BLOCK_SIZE,
                                temperature=1.0, device=DEVICE, greedy=True)
    jobs = [(proposals[b], list(encoded_goals[b][2])) for b in range(len(prompts))]
    verdicts = vpool.verify_batch(jobs)
    out = []
    for b, (label, _g, _gt, _ct) in enumerate(encoded_goals):
        out.append((label, len(proposals[b]), verdicts[b]))
    model.train()
    return out


# ---------------------------------------------------------------------------
# Checkpointing.
# ---------------------------------------------------------------------------
def save_ckpt(path, model, optim, round_idx):
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "round": round_idx,
        "config": {
            "n_layer": N_LAYER, "n_head": N_HEAD, "n_embd": N_EMBD,
            "block_size": BLOCK_SIZE, "max_gen": MAX_GEN,
            "samples_per_goal": SAMPLES_PER_GOAL,
            "temperature": TEMPERATURE,
            "upweight": UPWEIGHT, "lr": LR, "seed": SEED,
        },
    }, path)
    log(f"checkpoint saved at round {round_idx}")


def load_ckpt(path, model, optim):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optim"])
    return int(ckpt.get("round", 0))


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    log(f"=== microgpt_expit starting (pid={os.getpid()}) ===")
    log(f"DEVICE={DEVICE}  NUM_ROUNDS={NUM_ROUNDS}  SAMPLES_PER_GOAL={SAMPLES_PER_GOAL}")
    log(f"TEMPERATURE={TEMPERATURE}  SFT_BATCH={SFT_BATCH}  SFT_STEPS={SFT_STEPS}  "
        f"UPWEIGHT={UPWEIGHT}  LR={LR}")
    log(f"VERIFIER_THREADS={VERIFIER_THREADS}")
    log(f"arch: n_layer={N_LAYER} n_head={N_HEAD} n_embd={N_EMBD} head_dim={HEAD_DIM}")
    log(f"vocab={T.VOCAB_SIZE}  block={BLOCK_SIZE}  max_gen={MAX_GEN}")
    log(f"#seeds: {len(encoded_goals)}")
    for label, _g, gt, ct in encoded_goals:
        log(f"  {label}: prompt-toks={len(gt)} gold-cert-toks={len(ct)}")

    torch.manual_seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Buffer: load JSONL if present, else seed from corpus and persist.
    buffer, buffer_keys = load_buffer_jsonl(BUFFER_PATH)
    if not buffer:
        buffer, buffer_keys = init_buffer_from_corpus()
        append_buffer_jsonl(BUFFER_PATH, buffer)
        log(f"seeded buffer with {len(buffer)} corpus entries → {BUFFER_PATH}")
    else:
        log(f"loaded buffer with {len(buffer)} entries from {BUFFER_PATH}")

    # Model + optim.
    model = HOLGPT(T.VOCAB_SIZE, BLOCK_SIZE).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"num params: {n_params}")

    start_round = 0
    if os.path.exists(CKPT_PATH):
        start_round = load_ckpt(CKPT_PATH, model, optim)
        log(f"resumed from checkpoint at round {start_round}")

    vpool = VerifierPool(VERIFIER_THREADS, VERIFIER_BIN)

    try:
        # Round 0: bootstrap SFT over the initial buffer.
        if start_round == 0:
            t0 = time.time()
            log(f"=== Round 0 (bootstrap SFT) over {len(buffer)} entries, "
                f"{SFT_STEPS} steps ===")
            loss = run_sft_epoch(model, optim, buffer,
                                  batch=SFT_BATCH,
                                  weights=[1.0] * len(buffer),
                                  n_steps=SFT_STEPS)
            log(f"round 0 bootstrap SFT done: loss={loss:.4f}  ({time.time()-t0:.1f}s)")
            ev = greedy_eval(model, vpool)
            solved = sum(1 for (_l, _gl, v) in ev if v == 100)
            log(f"round 0 eval: solved={solved}/{len(encoded_goals)}  "
                f"{' '.join(f'{l}=V{v:+d}' for l, _gl, v in ev)}")
            save_ckpt(CKPT_PATH, model, optim, 0)
            start_round = 0  # next round will be 1

        # Rounds 1..NUM_ROUNDS.
        for round_idx in range(start_round + 1, NUM_ROUNDS + 1):
            t_round = time.time()
            log(f"=== Round {round_idx} ===")

            # 1) Exploration.
            t0 = time.time()
            prompts = []
            for _ in range(SAMPLES_PER_GOAL):
                for (_l, _g, gt, _ct) in encoded_goals:
                    prompts.append([T.BOS] + list(gt) + [T.BOS])
            proposals = sample_rollouts(model, prompts, MAX_GEN, BLOCK_SIZE,
                                        temperature=TEMPERATURE,
                                        device=DEVICE, greedy=False)
            t_sample = time.time() - t0

            # 2) Verification.
            t0 = time.time()
            jobs = []
            goal_for = []
            for i in range(len(prompts)):
                seed_i = i % len(encoded_goals)
                goal_toks = list(encoded_goals[seed_i][2])
                jobs.append((proposals[i], goal_toks))
                goal_for.append(seed_i)
            verdicts = vpool.verify_batch(jobs)
            t_verify = time.time() - t0

            # 3) Dedup + append.
            per_seed_success = [0] * len(encoded_goals)
            new_entries: List[BufferEntry] = []
            for i, v in enumerate(verdicts):
                if v != 100:
                    continue
                per_seed_success[goal_for[i]] += 1
                e = BufferEntry(
                    goal_toks=jobs[i][1],
                    cert_toks=list(proposals[i]),
                    source="discovered",
                    discovered_round=round_idx,
                )
                k = _key(e)
                if k in buffer_keys:
                    continue
                buffer_keys.add(k)
                buffer.append(e)
                new_entries.append(e)
            if new_entries:
                append_buffer_jsonl(BUFFER_PATH, new_entries)
            log(f"  exploration: {len(prompts)} proposals, "
                f"{sum(per_seed_success)} verified (+100), "
                f"{len(new_entries)} new unique → buffer={len(buffer)} "
                f"({t_sample:.1f}s sample / {t_verify:.1f}s verify)")
            for s_i, (lbl, _g, _gt, _ct) in enumerate(encoded_goals):
                if per_seed_success[s_i] > 0:
                    log(f"    {lbl}: {per_seed_success[s_i]}/{SAMPLES_PER_GOAL} succeeded")

            # 4) SFT epoch with this round's new entries upweighted.
            t0 = time.time()
            weights = [
                UPWEIGHT if e.discovered_round == round_idx else 1.0
                for e in buffer
            ]
            sft_loss = run_sft_epoch(model, optim, buffer,
                                      batch=SFT_BATCH, weights=weights,
                                      n_steps=SFT_STEPS)
            t_sft = time.time() - t0
            log(f"  SFT epoch: loss={sft_loss:.4f}  ({t_sft:.1f}s)")

            # 5) Greedy eval.
            ev = greedy_eval(model, vpool)
            solved = sum(1 for (_l, _gl, v) in ev if v == 100)
            log(f"  eval: solved={solved}/{len(encoded_goals)}  "
                f"{' '.join(f'{l}=V{v:+d}' for l, _gl, v in ev)}")
            log(f"  round {round_idx} total: {time.time()-t_round:.1f}s")

            save_ckpt(CKPT_PATH, model, optim, round_idx)

        log("=== run complete ===")
    finally:
        vpool.close()
        _log_f.close()


if __name__ == "__main__":
    main()

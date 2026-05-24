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
from typing import List, Tuple, Optional, Set, Sequence

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
    SEEDS, encoded_goals, ENCODED_CORPUS, CORPUS,
)


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------
LOG_PATH         = os.environ.get("HOL_EXPIT_LOG_PATH",    os.path.join(HERE, "hol_expit.log"))
BUFFER_PATH      = os.environ.get("HOL_EXPIT_BUFFER_PATH", os.path.join(HERE, "hol_expit_buffer.jsonl"))
NOVEL_PATH       = os.environ.get("HOL_EXPIT_NOVEL_PATH",  os.path.join(HERE, "hol_expit_novel.jsonl"))
CKPT_PATH        = os.environ.get("HOL_EXPIT_CKPT_PATH",   os.path.join(HERE, "hol_expit_ckpt.pt"))
NUM_ROUNDS       = int(os.environ.get("HOL_EXPIT_NUM_ROUNDS",       "10"))
SAMPLES_PER_GOAL = int(os.environ.get("HOL_EXPIT_SAMPLES_PER_GOAL", "32"))
TEMPERATURE      = float(os.environ.get("HOL_EXPIT_TEMPERATURE",    "1.0"))
SFT_BATCH        = int(os.environ.get("HOL_EXPIT_SFT_BATCH",        "64"))
SFT_STEPS        = int(os.environ.get("HOL_EXPIT_SFT_STEPS",        "200"))
UPWEIGHT         = float(os.environ.get("HOL_EXPIT_UPWEIGHT",       "4.0"))
LR               = float(os.environ.get("HOL_EXPIT_LR",             "3e-4"))
BLOCK_SIZE       = int(os.environ.get("HOL_EXPIT_BLOCK_SIZE",       "320"))
MAX_GEN          = int(os.environ.get("HOL_EXPIT_MAX_GEN",          "256"))
VERIFIER_THREADS = int(os.environ.get("HOL_EXPIT_VERIFIER_THREADS",
                                       str(max(1, (os.cpu_count() or 4) - 4))))
SEED             = int(os.environ.get("HOL_EXPIT_SEED", "42"))
VERIFIER_BIN     = os.environ.get(
    "HOL_VERIFIER_BIN",
    os.path.join(HERE, "_build", "default", "bin", "verify_tokens.exe"),
)
# Continuous-training knobs.  When SYNTH_PER_ROUND > 0, the trainer
# generates this many synthetic-backward samples per round, appends them
# to the buffer JSONL, and includes them in subsequent SFT epochs.  This
# is the "continuously expanding corpus" loop: each round both
# (a) accepts model-discovered certs and (b) extends the corpus with
# new kernel-valid proof shapes that the generator stumbles into.
SYNTH_PER_ROUND  = int(os.environ.get("HOL_EXPIT_SYNTH_PER_ROUND", "0"))
SYNTH_MIN_DEPTH  = int(os.environ.get("HOL_EXPIT_SYNTH_MIN_DEPTH",  "2"))
SYNTH_MAX_DEPTH  = int(os.environ.get("HOL_EXPIT_SYNTH_MAX_DEPTH",  "8"))
SYNTH_VARIANTS   = int(os.environ.get("HOL_EXPIT_SYNTH_VARIANTS",   "10"))
# Temperature for the adaptive inverse-frequency rule sampler in
# synth/backward_gen.py.  T=2.0 is the empirically tuned default
# (pos-0 entropy 1.02, MK_COMB ~2.2% in a 1000-sample probe vs 0.18 /
# 1.2% baseline).  Lower T over-corrects toward BETA; higher T flattens
# the inverse-freq boost so rare-rule prevalence stops rising.
SYNTH_TEMP       = float(os.environ.get("HOL_EXPIT_SYNTH_TEMP",      "2.0"))
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
    """Append entries to the JSONL and fsync — so a crash never loses
    a verified discovery the trainer already accepted."""
    with open(path, "a") as f:
        for e in entries:
            f.write(json.dumps(asdict(e)) + "\n")
        f.flush()
        os.fsync(f.fileno())


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
# Novelty detection: a discovered cert is "novel" iff its rule-sequence
# shape never appears in any supervised-corpus cert.  This is the metric
# the experiment cares about — does ExitIt invent new rule combinations,
# or only amplify what was planted in training?
# ---------------------------------------------------------------------------
_RULE_NAME_BY_TOK = {tok: name for name, tok in T.RULE_TOK.items()}


def cert_rule_shape(cert_toks: Sequence[int]) -> Tuple[str, ...]:
    """Extract the ordered tuple of rule names used in a cert by scanning
    for `KW_rule <RULE>` patterns.  Independent of pool slots — only the
    rule sequence matters."""
    rules: List[str] = []
    n = len(cert_toks)
    for i in range(n - 1):
        if cert_toks[i] == T.KW_RULE:
            # Next rule-class token (skip parens/whitespace if any).
            for j in range(i + 1, min(i + 4, n)):
                t = cert_toks[j]
                if T.RULE_FIRST <= t <= T.RULE_LAST:
                    rules.append(_RULE_NAME_BY_TOK[t])
                    break
    return tuple(rules)


def _build_corpus_rule_shapes() -> set:
    """Pre-compute the set of rule-sequence shapes in the supervised
    corpus.  Anything not in here is a NEW rule sequence the model
    invented during exploration."""
    shapes: set = set()
    for goal, cert in CORPUS:
        cert_toks, _ = T.encode_cert(cert)
        shapes.add(cert_rule_shape(cert_toks))
    return shapes


CORPUS_RULE_SHAPES = _build_corpus_rule_shapes()


def append_novel_jsonl(path: str, entries):
    if not entries:
        return
    with open(path, "a") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
        f.flush()
        os.fsync(f.fileno())


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
    """Atomic checkpoint save: write to <path>.tmp then rename.
    Protects against a mid-write crash (machine reboot, OOM, kill -9)
    leaving a truncated .pt that can't be loaded on resume."""
    tmp = path + ".tmp"
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
    }, tmp)
    os.replace(tmp, path)
    log(f"checkpoint saved at round {round_idx}")


def load_ckpt(path, model, optim):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optim"])
    return int(ckpt.get("round", 0))


# ---------------------------------------------------------------------------
# Continuous corpus expansion.  Each round, generate K new synthetic-
# backward samples, append them to the buffer JSONL (durable across
# crashes), and add them to the in-memory buffer so the next SFT pass
# sees them.
# ---------------------------------------------------------------------------
def expand_corpus_synthetic(buffer: List[BufferEntry],
                              buffer_keys: set,
                              n_samples: int,
                              round_idx: int,
                              rng: random.Random) -> int:
    """Generate up to n_samples kernel-valid (goal, cert) pairs and
    append them as buffer entries.  Returns the number actually added
    (after dedup + length cap).  Skips if the synth package isn't
    importable so the trainer still works in environments without it.
    """
    if n_samples <= 0:
        return 0
    try:
        sys.path.insert(0, HERE)
        from synth.backward_gen import generate_one, encode_pair, variant_of, GeneratorState
        from synth.freq_counter import build_from_buffer
    except ImportError as e:
        log(f"synth unavailable: {e}")
        return 0

    # Build a position-aware rule-frequency snapshot from the persistent
    # buffer, then thread it through generate_one as an adaptive sampler
    # state.  Per-batch attempts/successes are tracked online; freq.bump
    # inside the generator gives within-round feedback so the 2000-sample
    # batch self-balances rather than the first sample dominating.
    t_freq = time.time()
    freq = build_from_buffer(BUFFER_PATH)
    gen_state = GeneratorState(freq=freq, temperature=SYNTH_TEMP)
    log(f"  synth sampler: freq snapshot in {time.time()-t_freq:.1f}s "
        f"({freq.total_proofs} proofs, T={SYNTH_TEMP})")

    added: List[BufferEntry] = []
    attempts = 0
    seen_now = set(buffer_keys)  # local copy for fast lookup
    while len(added) < n_samples and attempts < n_samples * 8:
        attempts += 1
        depth = rng.randint(SYNTH_MIN_DEPTH, SYNTH_MAX_DEPTH)
        gr = generate_one(rng, depth, state=gen_state)
        if gr is None:
            continue
        variants = [gr]
        for _ in range(SYNTH_VARIANTS - 1):
            v = variant_of(gr, rng)
            if v is not None:
                variants.append(v)
        for vr in variants:
            if len(added) >= n_samples:
                break
            try:
                cert_toks, goal_toks, _c_hdr, _g_hdr = encode_pair(vr)
            except Exception:
                continue
            if len(cert_toks) + len(goal_toks) + 3 > BLOCK_SIZE:
                continue
            key = (tuple(goal_toks), tuple(cert_toks))
            if key in seen_now:
                continue
            seen_now.add(key)
            entry = BufferEntry(
                goal_toks=list(goal_toks),
                cert_toks=list(cert_toks),
                source="synthetic",
                discovered_round=round_idx,
            )
            added.append(entry)

    if added:
        append_buffer_jsonl(BUFFER_PATH, added)
        buffer.extend(added)
        for e in added:
            buffer_keys.add(_key(e))
    return len(added)


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
        seed_shape = cert_rule_shape(ct)
        seed_novel = "" if seed_shape in CORPUS_RULE_SHAPES else "  [OOD shape]"
        log(f"  {label}: prompt-toks={len(gt)} gold-cert-toks={len(ct)}{seed_novel}")
    log(f"corpus rule-sequence baseline: {len(CORPUS_RULE_SHAPES)} unique shapes")
    for s in sorted(CORPUS_RULE_SHAPES):
        log(f"  baseline: {'-'.join(s) if s else '(empty)'}")

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
            novel_records = []           # for the novel JSONL
            novel_shapes_this_round = {} # shape -> sample (label, gen_len) for log
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
                # Novelty check: does this cert use a rule-sequence shape
                # absent from the supervised corpus?  If so, record it
                # — this is the metric for "did the model invent something".
                shape = cert_rule_shape(e.cert_toks)
                if shape and shape not in CORPUS_RULE_SHAPES:
                    seed_label = encoded_goals[goal_for[i]][0]
                    if shape not in novel_shapes_this_round:
                        novel_shapes_this_round[shape] = (seed_label, len(e.cert_toks))
                    novel_records.append({
                        "round": round_idx,
                        "seed_label": seed_label,
                        "shape": list(shape),
                        "goal_toks": e.goal_toks,
                        "cert_toks": e.cert_toks,
                    })
                k = _key(e)
                if k in buffer_keys:
                    continue
                buffer_keys.add(k)
                buffer.append(e)
                new_entries.append(e)
            if new_entries:
                append_buffer_jsonl(BUFFER_PATH, new_entries)
            if novel_records:
                append_novel_jsonl(NOVEL_PATH, novel_records)
            log(f"  exploration: {len(prompts)} proposals, "
                f"{sum(per_seed_success)} verified (+100), "
                f"{len(new_entries)} new unique → buffer={len(buffer)} "
                f"({t_sample:.1f}s sample / {t_verify:.1f}s verify)")

            # 3b) Continuous corpus expansion: append fresh synthetic-
            # backward samples to the buffer.  These extend the rule-
            # shape distribution the model sees in the next SFT pass
            # without requiring an offline corpus rebuild.
            if SYNTH_PER_ROUND > 0:
                t_syn = time.time()
                rng = random.Random(SEED + round_idx * 1009)
                n_added = expand_corpus_synthetic(
                    buffer, buffer_keys, SYNTH_PER_ROUND, round_idx, rng)
                log(f"  synthesised {n_added} new corpus samples "
                    f"({time.time() - t_syn:.1f}s) → buffer={len(buffer)}")
            if novel_shapes_this_round:
                log(f"  *** NOVEL rule sequences this round: {len(novel_shapes_this_round)} unique, {len(novel_records)} total ***")
                for shape, (lbl, n_toks) in novel_shapes_this_round.items():
                    log(f"      [{lbl}] {' -> '.join(shape)}  (cert={n_toks} toks)")
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

        # End-of-run novelty summary: scan the novel JSONL we wrote and
        # report which (if any) rule sequences the model invented that
        # the supervised corpus never showed.
        novel_shape_counts = {}
        novel_total = 0
        if os.path.exists(NOVEL_PATH):
            with open(NOVEL_PATH) as f:
                for line in f:
                    rec = json.loads(line)
                    novel_total += 1
                    shp = tuple(rec["shape"])
                    novel_shape_counts[shp] = novel_shape_counts.get(shp, 0) + 1
        log("=== Novelty summary ===")
        log(f"  corpus baseline shapes: {len(CORPUS_RULE_SHAPES)}")
        log(f"  novel certs discovered: {novel_total} across {len(novel_shape_counts)} unique shapes")
        for shp, cnt in sorted(novel_shape_counts.items(), key=lambda x: -x[1]):
            log(f"    {cnt:4d}× {' -> '.join(shp) if shp else '(empty)'}")
        log("=== run complete ===")
    finally:
        vpool.close()
        _log_f.close()


if __name__ == "__main__":
    main()

"""microgpt_with_RL_hol_gpu.py — GPU variant of the HOL REINFORCE trainer.

PyTorch port of microgpt_with_RL_hol.py. Same semantics — REINFORCE with
per-prompt EMA baselines, grammar-masked sampling, reward-gated entropy
bonus, supervised warmup, kernel-verifier reward — but everything model-side
runs on GPU 0, and the kernel verifier is parallelised across a
ThreadPoolExecutor of persistent OCaml subprocesses.

Architecture upgrade vs the CPU trainer (head_dim=3 was too narrow):
    n_layer=2, n_head=4, n_embd=64, head_dim=16    (~130K params)

Mixed-seed batches: every RL step samples K rollouts of all 7 seeds in one
batched forward pass (default K=12 → batch=84).

Run:

    HOL_GPU_NUM_STEPS=200 HOL_GPU_BATCH_K=12 HOL_GPU_WARMUP_STEPS=50 \\
      python3 microgpt_with_RL_hol_gpu.py
"""

import os
import sys
import time
import math
import random
import subprocess
import threading
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Tokenizer.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tokenizer", "python"))
import hol_tokenizer as T

# Reuse stateless data builders from the CPU trainer.  Importing this module
# opens hol_rl_run.log in append mode as a benign side effect (no writes
# unless log() is called there).
sys.path.insert(0, os.path.dirname(__file__))
from microgpt_with_RL_hol import (
    SEEDS, encoded_goals,
    _build_supervised_corpus, ENCODED_CORPUS,
)

# ---------------------------------------------------------------------------
# Config (env vars).
# ---------------------------------------------------------------------------
HERE             = os.path.dirname(os.path.abspath(__file__))
LOG_PATH         = os.environ.get("HOL_GPU_LOG_PATH",  os.path.join(HERE, "hol_rl_gpu.log"))
CKPT_PATH        = os.environ.get("HOL_GPU_CKPT_PATH", os.path.join(HERE, "hol_rl_gpu_ckpt.pt"))
NUM_STEPS        = int(os.environ.get("HOL_GPU_NUM_STEPS",  "4000"))
BATCH_K          = int(os.environ.get("HOL_GPU_BATCH_K",    "12"))
WARMUP_STEPS     = int(os.environ.get("HOL_GPU_WARMUP_STEPS","50"))
VERIFIER_THREADS = int(os.environ.get("HOL_GPU_VERIFIER_THREADS",
                                       str(max(1, (os.cpu_count() or 4) - 4))))
ENTROPY_BETA     = float(os.environ.get("HOL_GPU_ENTROPY_BETA", "0.0"))
LR               = float(os.environ.get("HOL_GPU_LR",          "3e-4"))
BLOCK_SIZE       = int(os.environ.get("HOL_GPU_BLOCK_SIZE",    "96"))
MAX_GEN          = int(os.environ.get("HOL_GPU_MAX_GEN",       "80"))
LOG_EVERY        = int(os.environ.get("HOL_GPU_LOG_EVERY",     "10"))
CKPT_EVERY       = int(os.environ.get("HOL_GPU_CKPT_EVERY",    "100"))
SEED             = int(os.environ.get("HOL_GPU_SEED", "42"))
WARMUP_BATCH     = int(os.environ.get("HOL_GPU_WARMUP_BATCH",  "64"))
VERIFIER_BIN     = os.environ.get(
    "HOL_VERIFIER_BIN",
    os.path.join(HERE, "_build", "default", "bin", "verify_tokens.exe"),
)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N_LAYER  = 2
N_HEAD   = 4
N_EMBD   = 64
HEAD_DIM = N_EMBD // N_HEAD
assert HEAD_DIM == 16

# ---------------------------------------------------------------------------
# Logging.
# ---------------------------------------------------------------------------
_log_f = open(LOG_PATH, "a", buffering=1)
def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    _log_f.write(line + "\n")

# ---------------------------------------------------------------------------
# Model.
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, C, eps=1e-5):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, C, H):
        super().__init__()
        assert C % H == 0
        self.C = C; self.H = H; self.d = C // H
        self.wq = nn.Linear(C, C, bias=False)
        self.wk = nn.Linear(C, C, bias=False)
        self.wv = nn.Linear(C, C, bias=False)
        self.wo = nn.Linear(C, C, bias=False)

    def forward(self, x):
        B, T_, C = x.shape
        q = self.wq(x).view(B, T_, self.H, self.d).transpose(1, 2)  # [B, H, T, d]
        k = self.wk(x).view(B, T_, self.H, self.d).transpose(1, 2)
        v = self.wv(x).view(B, T_, self.H, self.d).transpose(1, 2)
        # SDPA picks Flash/efficient kernel when available.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T_, C)
        return self.wo(y)


class MLP(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.fc1 = nn.Linear(C, 4 * C, bias=False)
        self.fc2 = nn.Linear(4 * C, C, bias=False)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, C, H):
        super().__init__()
        self.ln1 = RMSNorm(C); self.attn = CausalSelfAttention(C, H)
        self.ln2 = RMSNorm(C); self.mlp  = MLP(C)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class HOLGPT(nn.Module):
    def __init__(self, vocab_size, block_size,
                 n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.ln_in = RMSNorm(n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        # idx: [B, T] long.  Returns logits [B, T, V].
        B, T_ = idx.shape
        pos = torch.arange(T_, device=idx.device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)
        x = self.ln_in(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Verifier thread pool.  Each thread owns one persistent OCaml subprocess.
# ---------------------------------------------------------------------------
class VerifierPool:
    def __init__(self, n_threads: int, binary_path: str):
        self._bin = binary_path
        self._tls = threading.local()
        self._exec = ThreadPoolExecutor(
            max_workers=n_threads,
            initializer=self._init_thread,
        )
        self._n = n_threads

    def _spawn(self) -> subprocess.Popen:
        return subprocess.Popen(
            [self._bin],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=1, universal_newlines=True,
        )

    def _init_thread(self):
        self._tls.proc = self._spawn()

    def _verify_one(self, cert_toks: List[int], goal_toks: List[int]) -> int:
        proc: subprocess.Popen = self._tls.proc
        parts = [str(len(cert_toks))] + [str(t) for t in cert_toks] \
              + [str(len(goal_toks))] + [str(t) for t in goal_toks]
        req = " ".join(parts) + "\n"
        try:
            proc.stdin.write(req)
            proc.stdin.flush()
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("verifier EOF")
            return int(line.strip())
        except Exception:
            # Restart subprocess and return -1 for this request.
            try: proc.kill()
            except Exception: pass
            self._tls.proc = self._spawn()
            return -1

    def verify_batch(self, jobs) -> List[int]:
        futs = [self._exec.submit(self._verify_one, c, g) for (c, g) in jobs]
        return [f.result() for f in futs]

    def close(self):
        self._exec.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------
# Batched grammar-masked rollout.
# ---------------------------------------------------------------------------
@dataclass
class RolloutBatch:
    sampled_tokens: torch.Tensor      # [B, T_gen] long, no grad
    log_probs:     torch.Tensor       # [B, T_gen] float, requires grad
    entropies:     torch.Tensor       # [B, T_gen] float, requires grad
    live_mask:     torch.Tensor       # [B, T_gen] float (1 where rollout was live), no grad
    gen_lens:      List[int]          # number of tokens actually appended per rollout
    gen_toks_per_rollout: List[List[int]]   # host-side, for verifier handoff


def _empty_rollout_batch(B: int, device) -> RolloutBatch:
    z = torch.zeros((B, 0), device=device)
    return RolloutBatch(
        sampled_tokens=torch.zeros((B, 0), dtype=torch.long, device=device),
        log_probs=z, entropies=z, live_mask=z,
        gen_lens=[0] * B,
        gen_toks_per_rollout=[[] for _ in range(B)],
    )


def rollout_batch(model: HOLGPT,
                  prompts: List[List[int]],
                  max_gen: int,
                  block_size: int,
                  device: torch.device,
                  deterministic: bool = False) -> RolloutBatch:
    B = len(prompts)
    if B == 0:
        return _empty_rollout_batch(0, device)

    L = [len(p) for p in prompts]
    L_max = max(L)
    cur = torch.full((B, L_max), T.PAD, dtype=torch.long, device=device)
    for b, p in enumerate(prompts):
        cur[b, :L[b]] = torch.tensor(p, dtype=torch.long, device=device)

    grammar = [T.initial_state() for _ in range(B)]
    live = [True] * B
    pos_next = list(L)                      # next predict position per rollout
    gen_toks_h: List[List[int]] = [[] for _ in range(B)]

    lp_steps, ent_steps, samp_steps, live_steps = [], [], [], []

    arange_B = torch.arange(B, device=device)

    for _t in range(max_gen):
        if cur.shape[1] >= block_size: break
        if not any(live): break

        logits_all = model(cur)                          # [B, T_cur, V], with grad
        idx_last = torch.tensor(
            [max(0, pn - 1) for pn in pos_next],
            dtype=torch.long, device=device,
        )
        logits = logits_all[arange_B, idx_last]          # [B, V]

        # Per-sequence grammar mask on CPU; dead rollouts get a "BOS-only" mask
        # so multinomial doesn't NaN.  Their lp gets zeroed by live_t below.
        mask_rows = []
        for b in range(B):
            if live[b]:
                mask_rows.append(T.valid_next_mask(grammar[b]))
            else:
                row = [False] * T.VOCAB_SIZE
                row[T.PAD] = True
                mask_rows.append(row)
        mask = torch.tensor(mask_rows, dtype=torch.bool, device=device)  # [B, V]

        masked = logits.masked_fill(~mask, float("-inf"))
        log_probs_step = F.log_softmax(masked, dim=-1)   # [B, V], with grad
        probs_step = log_probs_step.exp()

        if deterministic:
            sampled = probs_step.argmax(dim=-1)          # [B]
        else:
            # Guard against any all-False mask row by routing to PAD.
            row_ok = mask.any(dim=1)
            if not bool(row_ok.all().item()):
                safe = probs_step.detach().clone()
                safe[~row_ok] = 0.0
                safe[~row_ok, T.PAD] = 1.0
                sampled = torch.multinomial(safe, num_samples=1).squeeze(-1)
            else:
                sampled = torch.multinomial(probs_step.detach(), num_samples=1).squeeze(-1)

        lp = log_probs_step.gather(1, sampled.unsqueeze(1)).squeeze(1)         # [B]
        ent = -(probs_step * probs_step.clamp_min(1e-30).log()).sum(dim=-1)    # [B]

        live_t = torch.tensor(live, dtype=lp.dtype, device=device)
        lp = lp * live_t
        ent = ent * live_t

        # Advance PDA per-rollout on CPU.
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

        lp_steps.append(lp)
        ent_steps.append(ent)
        samp_steps.append(sampled)
        live_steps.append(live_t)

        cur = torch.cat([cur, sampled.unsqueeze(1)], dim=1)
        for b in range(B):
            # Each rollout's "next predict" is at the new last column.
            pos_next[b] = cur.shape[1]

    if not lp_steps:
        return _empty_rollout_batch(B, device)

    log_probs = torch.stack(lp_steps, dim=1)       # [B, T_gen]
    entropies = torch.stack(ent_steps, dim=1)
    sampled_tokens = torch.stack(samp_steps, dim=1)
    live_mask = torch.stack(live_steps, dim=1)

    return RolloutBatch(
        sampled_tokens=sampled_tokens,
        log_probs=log_probs,
        entropies=entropies,
        live_mask=live_mask,
        gen_lens=[len(g) for g in gen_toks_h],
        gen_toks_per_rollout=gen_toks_h,
    )


# ---------------------------------------------------------------------------
# Supervised warmup (batched teacher-forced CE).
# ---------------------------------------------------------------------------
def warmup_phase(model: HOLGPT, optim, corpus, n_steps: int, batch: int = WARMUP_BATCH):
    rng = random.Random(13)
    for step in range(n_steps):
        picks = [corpus[rng.randrange(len(corpus))] for _ in range(batch)]
        full_seqs = []
        prompt_lens = []
        for goal_toks, cert_toks in picks:
            prompt = [T.BOS] + list(goal_toks) + [T.BOS]
            target = list(cert_toks) + [T.EOS]
            full_seqs.append(prompt + target)
            prompt_lens.append(len(prompt))
        T_full = min(BLOCK_SIZE, max(len(f) for f in full_seqs))
        x = torch.full((batch, T_full), T.PAD, dtype=torch.long, device=DEVICE)
        for b, f in enumerate(full_seqs):
            ff = f[:T_full]
            x[b, :len(ff)] = torch.tensor(ff, dtype=torch.long, device=DEVICE)
        logits = model(x[:, :-1])                          # [B, T-1, V]
        targets = x[:, 1:].clone()                         # [B, T-1]
        # Ignore prediction positions that correspond to prompt tokens.
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
        if (step + 1) % max(1, n_steps // 20) == 0:
            log(f"  warmup {step+1:4d}/{n_steps}  loss={loss.item():.4f}")


# ---------------------------------------------------------------------------
# RL phase.
# ---------------------------------------------------------------------------
def build_rl_batch(K: int):
    """Returns (prompts, labels, goal_toks_per_rollout).  Each of the 7 seeds
    appears K times, in seed-major order."""
    prompts = []
    labels = []
    goal_toks_per_rollout = []
    for (label, _goal, gt, _ct) in encoded_goals:
        prompt = [T.BOS] + list(gt) + [T.BOS]
        for _ in range(K):
            prompts.append(prompt)
            labels.append(label)
            goal_toks_per_rollout.append(list(gt))
    return prompts, labels, goal_toks_per_rollout


def rl_phase(model: HOLGPT, optim, vpool: VerifierPool, baselines, start_step: int):
    t_start = time.time()
    last_log_t = t_start
    for step in range(start_step, NUM_STEPS):
        torch.cuda.reset_peak_memory_stats() if DEVICE.type == "cuda" else None
        prompts, labels, goal_toks = build_rl_batch(BATCH_K)
        B = len(prompts)

        batch = rollout_batch(model, prompts, MAX_GEN, BLOCK_SIZE, DEVICE,
                              deterministic=False)

        jobs = [(batch.gen_toks_per_rollout[b], goal_toks[b]) for b in range(B)]
        rewards = vpool.verify_batch(jobs)                         # list[int]

        # Per-rollout advantages.
        adv_list = [rewards[b] - baselines[labels[b]] for b in range(B)]
        adv = torch.tensor(adv_list, dtype=torch.float32, device=DEVICE)
        eff_beta = torch.tensor(
            [ENTROPY_BETA if rewards[b] < 50.0 else 0.0 for b in range(B)],
            dtype=torch.float32, device=DEVICE,
        )

        # The token-level multiplications happened at sample time, so live_mask
        # here is redundant for log_probs/entropies but kept for clarity.
        lp_sum  = (batch.log_probs * batch.live_mask).sum(dim=1)
        ent_sum = (batch.entropies * batch.live_mask).sum(dim=1)

        if batch.log_probs.shape[1] == 0:
            # No tokens generated at all (shouldn't happen).  Skip update.
            loss = torch.tensor(0.0, device=DEVICE)
        else:
            loss_pg = -(adv * lp_sum).mean()
            loss_en = -(eff_beta * ent_sum).mean()
            loss = loss_pg + loss_en
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        # EMA baseline update (per label, using this step's average reward).
        # All seeds appear K times in seed-major order, so slicing is cheap.
        for li, (lbl, _g, _gt, _ct) in enumerate(encoded_goals):
            chunk = rewards[li * BATCH_K : (li + 1) * BATCH_K]
            avg = sum(chunk) / len(chunk)
            baselines[lbl] = 0.9 * baselines[lbl] + 0.1 * avg

        if (step + 1) % LOG_EVERY == 0:
            now = time.time()
            rate = LOG_EVERY / (now - last_log_t) if now > last_log_t else 0.0
            last_log_t = now
            per_seed = []
            for li, (lbl, _g, _gt, _ct) in enumerate(encoded_goals):
                chunk = rewards[li * BATCH_K : (li + 1) * BATCH_K]
                per_seed.append(f"{lbl}={sum(chunk)/len(chunk):+.1f}")
            avg_b = sum(baselines.values()) / len(baselines)
            mem_mb = (torch.cuda.max_memory_allocated() / 1e6) if DEVICE.type == "cuda" else 0.0
            log(f"step {step+1:5d}/{NUM_STEPS} | loss {loss.item():+.3f} | "
                f"avg-b {avg_b:+6.2f} | mem {mem_mb:.1f}MB | {rate:.2f} steps/s | "
                f"{' '.join(per_seed)}")

        if (step + 1) % CKPT_EVERY == 0:
            save_ckpt(CKPT_PATH, model, optim, baselines, step + 1)


# ---------------------------------------------------------------------------
# Inference (greedy, grammar-masked) — same kernel verifier as RL.
# ---------------------------------------------------------------------------
def final_inference(model: HOLGPT, vpool: VerifierPool):
    model.eval()
    prompts = []
    goal_toks_list = []
    labels = []
    for (label, _goal, gt, _ct) in encoded_goals:
        prompts.append([T.BOS] + list(gt) + [T.BOS])
        goal_toks_list.append(list(gt))
        labels.append(label)
    with torch.no_grad():
        batch = rollout_batch(model, prompts, MAX_GEN, BLOCK_SIZE, DEVICE,
                              deterministic=True)
    jobs = [(batch.gen_toks_per_rollout[b], goal_toks_list[b])
            for b in range(len(prompts))]
    verdicts = vpool.verify_batch(jobs)
    log("=== Inference (greedy, kernel-verified) ===")
    for b, lbl in enumerate(labels):
        log(f"  {lbl:14s} gen-len {batch.gen_lens[b]:3d}  V={verdicts[b]:+d}")
    model.train()
    return verdicts


# ---------------------------------------------------------------------------
# Checkpointing.
# ---------------------------------------------------------------------------
def save_ckpt(path, model, optim, baselines, step):
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "baselines": dict(baselines),
        "step": step,
        "config": {
            "n_layer": N_LAYER, "n_head": N_HEAD, "n_embd": N_EMBD,
            "block_size": BLOCK_SIZE, "max_gen": MAX_GEN,
            "batch_k": BATCH_K, "lr": LR, "entropy_beta": ENTROPY_BETA,
            "seed": SEED,
        },
    }, path)
    log(f"checkpoint saved at step {step}")


def load_ckpt(path, model, optim):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optim"])
    return ckpt["baselines"], ckpt["step"]


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    log(f"=== microgpt_with_RL_hol_gpu starting (pid={os.getpid()}) ===")
    log(f"DEVICE={DEVICE}  NUM_STEPS={NUM_STEPS}  BATCH_K={BATCH_K}  WARMUP_STEPS={WARMUP_STEPS}")
    log(f"VERIFIER_THREADS={VERIFIER_THREADS}  ENTROPY_BETA={ENTROPY_BETA}  LR={LR}")
    log(f"arch: n_layer={N_LAYER} n_head={N_HEAD} n_embd={N_EMBD} head_dim={HEAD_DIM}")
    log(f"vocab={T.VOCAB_SIZE}  block={BLOCK_SIZE}  max_gen={MAX_GEN}")
    log(f"#seeds: {len(encoded_goals)}")
    for label, _g, gt, ct in encoded_goals:
        log(f"  {label}: prompt-toks={len(gt)} gold-cert-toks={len(ct)}")
    log(f"corpus size: {len(ENCODED_CORPUS)}")

    torch.manual_seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = HOLGPT(T.VOCAB_SIZE, BLOCK_SIZE).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"num params: {n_params}")

    baselines = {label: 0.0 for (label, _g, _gt, _ct) in encoded_goals}
    start_step = 0
    if os.path.exists(CKPT_PATH):
        bl, start_step = load_ckpt(CKPT_PATH, model, optim)
        baselines.update(bl)
        log(f"resumed from checkpoint at step {start_step}")

    vpool = VerifierPool(VERIFIER_THREADS, VERIFIER_BIN)

    try:
        if WARMUP_STEPS > 0 and start_step == 0:
            log(f"=== Warmup: {WARMUP_STEPS} supervised steps "
                f"over {len(ENCODED_CORPUS)} corpus examples ===")
            t0 = time.time()
            warmup_phase(model, optim, ENCODED_CORPUS, WARMUP_STEPS, batch=WARMUP_BATCH)
            log(f"=== Warmup done ({time.time()-t0:.1f}s) ===")

        rl_phase(model, optim, vpool, baselines, start_step)
        final_inference(model, vpool)
        save_ckpt(CKPT_PATH, model, optim, baselines, NUM_STEPS)
        log("=== run complete ===")
    finally:
        vpool.close()
        _log_f.close()


if __name__ == "__main__":
    main()

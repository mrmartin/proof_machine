"""microgpt_with_RL_hol.py — train a small GPT to find HOL proofs via REINFORCE.

PyTorch port of the original scalar-autograd trainer.  Single-GPU
training; rollouts run as a batched forward on one CUDA device with the
kernel verifier as a CPU subprocess pool.  The RL recipe — REINFORCE +
per-prompt EMA baseline + reward-gated entropy + grammar-masked sampling
+ kernel-verified reward + optional supervised warmup — is preserved
from the original.

Reward shape: integer codes from the OCaml verifier (-1 / 0 / 100) are
remapped to floats (`HOL_REWARD_GRAMMAR_REJECT / _WRONG_CONCL / _CORRECT`,
defaults -1.0 / 0.0 / 1000.0) so "proves the prompted theorem" dominates
"kernel-verifies but proves something else" by a 1000-unit margin.  The
advantage is normalised by `REWARD_CORRECT` before the policy-gradient
loss so the wider gap doesn't blow up the gradient magnitude.
"""

import os
import sys
import time
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tokenizer", "python"))
import hol_tokenizer as T

# ---------------------------------------------------------------------------
# Configuration via env vars (names preserved from the original).
# ---------------------------------------------------------------------------
LOG_PATH       = os.environ.get("HOL_LOG_PATH",  os.path.join(os.path.dirname(__file__), "hol_rl_run.log"))
CKPT_PATH      = os.environ.get("HOL_CKPT_PATH", os.path.join(os.path.dirname(__file__), "hol_rl_ckpt.pt"))
NUM_STEPS      = int(os.environ.get("HOL_NUM_STEPS",  "8000"))
LOG_EVERY      = int(os.environ.get("HOL_LOG_EVERY",  "20"))
CKPT_EVERY     = int(os.environ.get("HOL_CKPT_EVERY", "200"))
WARMUP_STEPS   = int(os.environ.get("HOL_WARMUP_STEPS", "0"))
ENTROPY_BETA   = float(os.environ.get("HOL_ENTROPY_BETA", "0.0"))
USE_KERNEL_VERIFY = int(os.environ.get("HOL_USE_KERNEL_VERIFY", "1"))
VERIFIER_BIN   = os.environ.get(
    "HOL_VERIFIER_BIN",
    os.path.join(os.path.dirname(__file__), "_build", "default", "bin", "verify_tokens.exe"),
)

# New: reward scale (the gap between "proved the theorem" and "verified
# the wrong theorem" used to be 100; now 1000 by default).
REWARD_GRAMMAR_REJECT = float(os.environ.get("HOL_REWARD_GRAMMAR_REJECT", "-1.0"))
REWARD_WRONG_CONCL    = float(os.environ.get("HOL_REWARD_WRONG_CONCL",    "0.0"))
REWARD_CORRECT        = float(os.environ.get("HOL_REWARD_CORRECT",        "1000.0"))
STEP_BONUS            = float(os.environ.get("HOL_STEP_BONUS", "0"))   # python-V fallback only

# Rollout / batching.  HOL_NUM_WORKERS = rollouts per Adam step (single
# batched forward).  Bigger amortises kernel-launch overhead.
N_ROLLOUTS_PER_STEP = int(os.environ.get("HOL_NUM_WORKERS", "128"))
max_gen_tokens      = int(os.environ.get("HOL_MAX_GEN", "80"))
block_size          = int(os.environ.get("HOL_BLOCK_SIZE", "128"))

# Model size (new envs; default 4 layers / 128 dim ≈ 850K params).
n_layer = int(os.environ.get("HOL_N_LAYER", "4"))
n_embd  = int(os.environ.get("HOL_N_EMBD",  "128"))
n_head  = int(os.environ.get("HOL_N_HEAD",  "4"))
assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
head_dim = n_embd // n_head
vocab_size = T.VOCAB_SIZE
BOS = T.BOS
EOS = T.EOS

LR = float(os.environ.get("HOL_LR", "1e-3"))

# Which CUDA device to use (defaults to 0).
CUDA_DEVICE = int(os.environ.get("HOL_CUDA_DEVICE", "0"))

# ---------------------------------------------------------------------------
# Logging.
# ---------------------------------------------------------------------------
log_f = None
def open_log():
    global log_f
    log_f = open(LOG_PATH, "a", buffering=1)

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_f is not None:
        log_f.write(line + "\n")

# ---------------------------------------------------------------------------
# Device setup.
# ---------------------------------------------------------------------------
def setup_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    torch.cuda.set_device(CUDA_DEVICE)
    return torch.device(f"cuda:{CUDA_DEVICE}")

# ---------------------------------------------------------------------------
# Seeds (preserved from the original trainer).
# ---------------------------------------------------------------------------
NAT  = T.nat_ty()
BOOL = T.bool_ty()

def G_refl(name):      v = T.mk_var(name, NAT);  return T.mk_eq(v, v)
def G_refl_bool(name): v = T.mk_var(name, BOOL); return T.mk_eq(v, v)
def G_imp_self(name):  p = T.mk_var(name, BOOL); return T.mk_imp(p, p)

SEEDS = [
    ("refl_x",      G_refl("x"),       1),
    ("refl_y",      G_refl("y"),       1),
    ("refl_n",      G_refl("n"),       1),
    ("refl_p_bool", G_refl_bool("p"),  1),
    ("refl_q_bool", G_refl_bool("q"),  1),
    ("imp_p",       G_imp_self("p"),   2),
    ("imp_q",       G_imp_self("q"),   2),
]

def gold_cert_for(label, goal):
    if label.startswith("refl_"):
        x = goal[2]
        return T.Cert(
            steps=[T.Step(1, "REFL", ("term", x), [])],
            concl=goal,
        )
    if label.startswith("imp_"):
        p = goal[2]
        return T.Cert(
            steps=[
                T.Step(1, "ASSUME", ("term", p), []),
                T.Step(2, "DISCH",  ("term", p), [1]),
            ],
            concl=goal,
        )
    raise ValueError(f"no gold cert for {label}")

# ---------------------------------------------------------------------------
# Supervised corpus (preserved verbatim from the original).
# ---------------------------------------------------------------------------
def _build_supervised_corpus():
    nat   = NAT
    bool_ = BOOL
    ind   = T.ind_ty()
    out = []

    def refl_pair(name, ty):
        v = T.mk_var(name, ty)
        return (T.mk_eq(v, v),
                T.Cert(steps=[T.Step(1, "REFL", ("term", v), [])],
                       concl=T.mk_eq(v, v)))

    def imp_pair(name):
        p = T.mk_var(name, bool_)
        return (T.mk_imp(p, p),
                T.Cert(steps=[
                    T.Step(1, "ASSUME", ("term", p), []),
                    T.Step(2, "DISCH",  ("term", p), [1]),
                ], concl=T.mk_imp(p, p)))

    def forall_refl_pair(name, ty):
        v = T.mk_var(name, ty)
        eq = T.mk_eq(v, v)
        return (T.mk_forall(name, ty, eq),
                T.Cert(steps=[
                    T.Step(1, "REFL", ("term", v), []),
                    T.Step(2, "GEN",  ("var", name, ty), [1]),
                ], concl=T.mk_forall(name, ty, eq)))

    def beta_pair(bound_name, witness_name):
        ty = nat
        body = T.mk_var(bound_name, ty)
        lam  = T.mk_abs(bound_name, ty, body)
        wit  = T.mk_var(witness_name, ty)
        app  = T.mk_comb(lam, wit)
        return (T.mk_eq(app, wit),
                T.Cert(steps=[T.Step(1, "BETA", ("term", app), [])],
                       concl=T.mk_eq(app, wit)))

    NAMES = ["a", "b", "c", "p", "q", "x", "y", "z", "n", "m", "k", "u", "v", "w"]
    per_pattern = int(os.environ.get("HOL_CORPUS_PER_PATTERN", "200"))

    for ty in (nat, bool_, ind):
        for _ in range(per_pattern):
            out.append(refl_pair(NAMES[len(out) % len(NAMES)], ty))

    for _ in range(per_pattern * 2):
        out.append(imp_pair(NAMES[len(out) % len(NAMES)]))

    for ty in (nat, bool_):
        for _ in range(per_pattern):
            out.append(forall_refl_pair(NAMES[len(out) % len(NAMES)], ty))

    for _ in range(per_pattern):
        out.append(beta_pair("x", NAMES[len(out) % len(NAMES)]))

    random.Random(42).shuffle(out)
    return out

CORPUS = _build_supervised_corpus()

ENCODED_CORPUS = []
for g, c in CORPUS:
    gt, _ = T.encode_term_only(g)
    ct, _ = T.encode_cert(c)
    if len(gt) + len(ct) + 3 <= block_size:
        ENCODED_CORPUS.append((gt, ct))

encoded_goals = []
for label, goal, _ in SEEDS:
    goal_toks, _ = T.encode_term_only(goal)
    gold = gold_cert_for(label, goal)
    cert_toks, _ = T.encode_cert(gold)
    encoded_goals.append((label, goal, goal_toks, cert_toks))

# ---------------------------------------------------------------------------
# Validator helpers (preserved verbatim).
# ---------------------------------------------------------------------------
def _infer_header(toks):
    hdr = T.PoolHeader()
    seen_v, seen_n, seen_tc, seen_tv = set(), set(), set(), set()
    for t in toks:
        if T.is_var(t):
            k = t - T.VAR_FIRST
            if k not in seen_v:
                seen_v.add(k)
                while len(hdr.vars) <= k: hdr.vars.append(f"v{len(hdr.vars)}")
        elif T.is_name(t):
            k = t - T.NAME_FIRST
            if k not in seen_n:
                seen_n.add(k)
                while len(hdr.names) <= k: hdr.names.append(f"n{len(hdr.names)}")
        elif T.is_tycon(t):
            k = t - T.TYCON_FIRST
            if k not in seen_tc:
                seen_tc.add(k)
                while len(hdr.tycons) <= k: hdr.tycons.append(f"t{len(hdr.tycons)}")
        elif T.is_tyvar(t):
            k = t - T.TYVAR_FIRST
            if k not in seen_tv:
                seen_tv.add(k)
                while len(hdr.tyvars) <= k: hdr.tyvars.append(f"a{len(hdr.tyvars)}")
    return hdr

def has_step_block(toks):
    for i, t in enumerate(toks):
        if t == T.KW_STEP and i > 0 and toks[i - 1] == T.LPAREN:
            return True
    return False

# ---------------------------------------------------------------------------
# PDA memoisation.
#
# The grammar PDA has a small number of reachable states (~ low hundreds).
# Without caching, `T.valid_next_mask` allocates a fresh [False]*232 list
# and walks the stack on EVERY call — ~10 µs each.  At B=32 rollouts × ~40
# generation steps that's >10 ms of Python work per RL step on a single
# core, which leaves the GPU idle between forwards.  Memoising both
# `valid_next_mask` and `T.step` turns those into dict lookups (~100 ns)
# after the first few rollouts warm the cache.
# ---------------------------------------------------------------------------
_ALL_TRUE_NP = np.ones(T.VOCAB_SIZE, dtype=bool)

_mask_cache: dict = {}
def cached_valid_next_mask_np(state):
    key = tuple(state)
    m = _mask_cache.get(key)
    if m is None:
        m = np.asarray(T.valid_next_mask(state), dtype=bool)
        _mask_cache[key] = m
    return m

_NO_SUCH = object()
_step_cache: dict = {}
def cached_step(state, tok):
    key = (tuple(state), tok)
    ns = _step_cache.get(key, _NO_SUCH)
    if ns is _NO_SUCH:
        ns = T.step(state, tok)
        _step_cache[key] = ns
    return ns

_accept_cache: dict = {}
def cached_is_accepting(state):
    key = tuple(state)
    v = _accept_cache.get(key)
    if v is None:
        v = bool(T.is_accepting(state))
        _accept_cache[key] = v
    return v

# ---------------------------------------------------------------------------
# Reward remap: verifier code -> float.
# ---------------------------------------------------------------------------
def remap_reward(code: int) -> float:
    if code == 100: return REWARD_CORRECT
    if code == 0:   return REWARD_WRONG_CONCL
    return REWARD_GRAMMAR_REJECT  # -1 or any other error

def V_python(goal, gen_tokens) -> float:
    """Pure-Python fallback validator, used when the kernel verifier
    subprocess is unavailable.  Returns the same remapped reward scale."""
    s = T.initial_state()
    for t in gen_tokens:
        ns = T.step(s, t)
        if ns is None: return REWARD_GRAMMAR_REJECT
        s = ns
    if not T.is_accepting(s):
        return REWARD_GRAMMAR_REJECT
    bonus = STEP_BONUS if has_step_block(gen_tokens) else 0.0
    try:
        hdr = _infer_header(gen_tokens)
        cert = T.decode_cert(hdr, gen_tokens)
    except Exception:
        return REWARD_WRONG_CONCL + bonus
    if T.alpha_eq(cert.concl, goal):
        return REWARD_CORRECT + bonus
    return REWARD_WRONG_CONCL + bonus

# ---------------------------------------------------------------------------
# Kernel verifier subprocess pool.
# ---------------------------------------------------------------------------
class Verifier:
    def __init__(self, bin_path):
        self.bin_path = bin_path
        self.p = subprocess.Popen(
            [bin_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=1, universal_newlines=True,
        )

    def score(self, cert_toks, goal_toks) -> int:
        try:
            parts = [str(len(cert_toks))] + [str(t) for t in cert_toks]
            parts += [str(len(goal_toks))] + [str(t) for t in goal_toks]
            req = " ".join(parts) + "\n"
            self.p.stdin.write(req)
            self.p.stdin.flush()
            line = self.p.stdout.readline()
            if not line:
                return -1
            return int(line.strip())
        except Exception:
            return -1

    def close(self):
        try:
            if self.p.stdin: self.p.stdin.close()
            self.p.terminate()
            self.p.wait(timeout=2.0)
        except Exception:
            try: self.p.kill()
            except Exception: pass

class VerifierPool:
    def __init__(self, n, bin_path):
        self.n = max(1, n)
        self.verifiers = [Verifier(bin_path) for _ in range(self.n)]
        self.exec = ThreadPoolExecutor(max_workers=self.n)

    def score_many(self, jobs):
        """jobs: list of (cert_toks, goal_toks).  Returns list of int codes
        in the same order.  Each verifier subprocess is owned by exactly
        one thread for the duration of the call so stdin/stdout reads
        never interleave on the same pipe."""
        n = self.n
        buckets = [[] for _ in range(n)]
        for i, (c, g) in enumerate(jobs):
            buckets[i % n].append((i, c, g))
        results = [0] * len(jobs)

        def run_bucket(v_idx):
            v = self.verifiers[v_idx]
            for orig_i, c, g in buckets[v_idx]:
                results[orig_i] = v.score(c, g)

        futs = [self.exec.submit(run_bucket, v_idx) for v_idx in range(n)]
        for f in futs: f.result()
        return results

    def close(self):
        for v in self.verifiers:
            v.close()
        self.exec.shutdown(wait=False)

# ---------------------------------------------------------------------------
# Model.
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        ms = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps) * self.weight

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, kv_cache=None):
        B, Tlen, C = x.shape
        q = self.wq(x).view(B, Tlen, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, Tlen, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, Tlen, self.n_head, self.head_dim).transpose(1, 2)
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        is_causal = (kv_cache is None and Tlen > 1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(B, Tlen, C)
        return self.wo(out), (k, v)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = RMSNorm(n_embd)
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x, kv_cache=None):
        a, new_kv = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + a
        h = self.fc1(self.ln2(x))
        h = F.relu(h)
        x = x + self.fc2(h)
        return x, new_kv

class MicroGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        self.n_layer = n_layer
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, ids, kv_caches=None, start_pos=0):
        """Returns (logits_last (B, V), new_kv_caches).
        - prefill: ids=(B, P), kv_caches=None, start_pos=0.
        - step:    ids=(B, 1), kv_caches=prev list, start_pos=position of ids."""
        B, Tlen = ids.shape
        pos = torch.arange(start_pos, start_pos + Tlen, device=ids.device)
        x = self.wte(ids) + self.wpe(pos)[None, :, :]
        new_caches = []
        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(x, kv_cache=kv)
            new_caches.append(new_kv)
        x = self.ln_f(x)
        return self.lm_head(x[:, -1, :]), new_caches

    def forward_seq(self, ids):
        """Full-sequence forward returning logits at every position
        (used for supervised warmup CE)."""
        B, Tlen = ids.shape
        pos = torch.arange(0, Tlen, device=ids.device)
        x = self.wte(ids) + self.wpe(pos)[None, :, :]
        for block in self.blocks:
            x, _ = block(x, kv_cache=None)
        x = self.ln_f(x)
        return self.lm_head(x)

# ---------------------------------------------------------------------------
# Rollout: B grammar-masked autoregressive samples on the same prompt.
# ---------------------------------------------------------------------------
def rollout(model, prompt_toks, B, device):
    V = vocab_size
    prompt_ids = torch.tensor(prompt_toks, dtype=torch.long, device=device)
    P = prompt_ids.shape[0]
    if P >= block_size:
        return (torch.zeros(B, device=device), torch.zeros(B, device=device),
                [[] for _ in range(B)])
    ids = prompt_ids.unsqueeze(0).expand(B, -1).contiguous()
    logits_last, kv = model(ids, kv_caches=None, start_pos=0)

    states = [T.initial_state() for _ in range(B)]
    done = [False] * B
    logp_sum = torch.zeros(B, device=device)
    ent_sum  = torch.zeros(B, device=device)
    generated = [[] for _ in range(B)]
    # Reusable CPU buffers to avoid per-step allocations.
    mask_np = np.empty((B, V), dtype=bool)
    live_np = np.zeros(B, dtype=np.float32)

    max_iters = min(max_gen_tokens, block_size - P)
    for t in range(max_iters):
        for i in range(B):
            if done[i]:
                mask_np[i] = _ALL_TRUE_NP
                live_np[i] = 0.0
                continue
            m = cached_valid_next_mask_np(states[i])
            if not m.any():
                done[i] = True
                mask_np[i] = _ALL_TRUE_NP
                live_np[i] = 0.0
            else:
                mask_np[i] = m
                live_np[i] = 1.0
        mask = torch.from_numpy(mask_np).to(device, non_blocking=True)
        live_t = torch.from_numpy(live_np).to(device, non_blocking=True)

        masked = logits_last.masked_fill(~mask, float('-inf'))
        log_p = F.log_softmax(masked, dim=-1)
        p     = log_p.exp()
        # Safe entropy: replace log_p with 0 at masked positions so the
        # backward through (p * log_p) doesn't see 0 * -inf = NaN.
        log_p_safe = torch.where(mask, log_p, torch.zeros_like(log_p))
        H     = -(p * log_p_safe).sum(-1)
        sampled = torch.multinomial(p.detach(), num_samples=1).squeeze(-1)
        logp_t  = log_p.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        logp_sum = logp_sum + logp_t * live_t
        ent_sum  = ent_sum  + H      * live_t

        sampled_cpu = sampled.tolist()
        for i in range(B):
            if done[i]:
                continue
            ns = cached_step(states[i], sampled_cpu[i])
            if ns is None:
                done[i] = True
                continue
            states[i] = ns
            generated[i].append(sampled_cpu[i])
            if cached_is_accepting(ns) or len(generated[i]) >= max_gen_tokens:
                done[i] = True
        if all(done):
            break
        if t + 1 >= max_iters:
            break
        logits_last, kv = model(sampled.unsqueeze(-1), kv_caches=kv, start_pos=P + t)

    return logp_sum, ent_sum, generated

# ---------------------------------------------------------------------------
# Greedy inference (end-of-run summary).
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_inference(model, device):
    model.eval()
    for label, goal, goal_toks, _ct in encoded_goals:
        prompt = [BOS] + list(goal_toks) + [BOS]
        prompt_ids = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
        P = prompt_ids.shape[1]
        if P >= block_size:
            log(f"  inference {label:14s} prompt too long")
            continue
        logits_last, kv = model(prompt_ids, kv_caches=None, start_pos=0)
        state = T.initial_state()
        out = []
        max_iters = min(max_gen_tokens, block_size - P)
        for t in range(max_iters):
            row = cached_valid_next_mask_np(state)
            mask = torch.from_numpy(row[None, :]).to(device, non_blocking=True)
            masked = logits_last.masked_fill(~mask, float('-inf'))
            sampled = int(masked.argmax(-1).item())
            ns = cached_step(state, sampled)
            if ns is None: break
            state = ns
            out.append(sampled)
            if cached_is_accepting(ns): break
            if t + 1 >= max_iters: break
            ids = torch.tensor([[sampled]], dtype=torch.long, device=device)
            logits_last, kv = model(ids, kv_caches=kv, start_pos=P + t)
        r = V_python(goal, out)
        log(f"  inference {label:14s} gen-len {len(out):3d} V={r:+.1f}")
    model.train()

# ---------------------------------------------------------------------------
# Supervised warmup.
# ---------------------------------------------------------------------------
class WarmupDataset(Dataset):
    def __init__(self, encoded_corpus, block_size):
        self.items = []
        for gt, ct in encoded_corpus:
            ids = [BOS] + list(gt) + [BOS] + list(ct) + [EOS]
            if len(ids) > block_size: continue
            prompt_end = 1 + len(gt) + 1  # index of the second BOS
            input_ids = ids[:-1]
            target_ids = ids[1:]
            mask = [1.0 if i >= (prompt_end - 1) else 0.0 for i in range(len(target_ids))]
            self.items.append((input_ids, target_ids, mask))
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def warmup_collate(batch):
    L = max(len(x[0]) for x in batch)
    inps = torch.zeros(len(batch), L, dtype=torch.long)
    tgts = torch.zeros(len(batch), L, dtype=torch.long)
    msks = torch.zeros(len(batch), L, dtype=torch.float)
    for i, (a, b, m) in enumerate(batch):
        n = len(a)
        inps[i, :n] = torch.tensor(a, dtype=torch.long)
        tgts[i, :n] = torch.tensor(b, dtype=torch.long)
        msks[i, :n] = torch.tensor(m, dtype=torch.float)
    return inps, tgts, msks

def run_warmup(model, optimizer, device):
    ds = WarmupDataset(ENCODED_CORPUS, block_size)
    if len(ds) == 0:
        log("warmup: empty dataset, skipping")
        return
    batch_size = max(1, min(32, len(ds)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=warmup_collate, num_workers=0,
                        pin_memory=True, drop_last=False)
    log(f"=== Warmup: {WARMUP_STEPS} steps over {len(ds)} examples (bs={batch_size}) ===")
    t_start = time.time()
    step_count = 0
    while step_count < WARMUP_STEPS:
        for inps, tgts, msks in loader:
            if step_count >= WARMUP_STEPS: break
            inps = inps.to(device, non_blocking=True)
            tgts = tgts.to(device, non_blocking=True)
            msks = msks.to(device, non_blocking=True)
            logits = model.forward_seq(inps)
            V_ = logits.shape[-1]
            ce = F.cross_entropy(logits.reshape(-1, V_), tgts.reshape(-1), reduction='none')
            ce = ce * msks.reshape(-1)
            denom = msks.sum().clamp_min(1.0)
            loss = ce.sum() / denom
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step_count += 1
            if step_count % max(1, WARMUP_STEPS // 20) == 0:
                log(f"  warmup {step_count:4d}/{WARMUP_STEPS}  avg-log-p={-loss.item():+.3f}")
    log(f"=== Warmup done ({time.time()-t_start:.1f}s) ===")

# ---------------------------------------------------------------------------
# Checkpoint.
# ---------------------------------------------------------------------------
def save_ckpt(path, model, optimizer, baselines, step):
    ckpt = {
        'step': step,
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'baselines': dict(baselines),
        'config': {
            'n_layer': n_layer, 'n_embd': n_embd, 'n_head': n_head,
            'block_size': block_size, 'vocab_size': vocab_size,
        },
    }
    torch.save(ckpt, path)
    log(f"checkpoint saved at step {step}")

def load_ckpt(path, model, optimizer, device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(ckpt, dict) or 'model' not in ckpt:
            raise ValueError("ckpt missing 'model' key (probably old scalar-Value pickle)")
    except Exception as e:
        log(f"ERROR: could not load {path}: {e}")
        log("Old scalar-Value pickles (R1-R5 hol_rl_r*.pkl) are not "
            "resumable by the PyTorch trainer. Rename or delete and start fresh.")
        raise
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    log(f"resumed from checkpoint at step {ckpt['step']}")
    return ckpt['step'], dict(ckpt['baselines'])

# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    device = setup_device()
    open_log()
    log(f"=== microgpt_with_RL_hol PyTorch port (pid={os.getpid()}) ===")
    log(f"device={device}")
    log(f"NUM_STEPS={NUM_STEPS}  LOG_EVERY={LOG_EVERY}  CKPT_EVERY={CKPT_EVERY}")
    log(f"WARMUP_STEPS={WARMUP_STEPS}  ENTROPY_BETA={ENTROPY_BETA}  LR={LR}")
    log(f"reward shape: reject={REWARD_GRAMMAR_REJECT}  wrong={REWARD_WRONG_CONCL}  correct={REWARD_CORRECT}")
    log(f"model: n_layer={n_layer} n_embd={n_embd} n_head={n_head} block_size={block_size}")
    log(f"vocab={vocab_size} max_gen={max_gen_tokens}  rollouts/step={N_ROLLOUTS_PER_STEP}")
    log(f"#seeds: {len(SEEDS)}")
    for label, _, gt, ct in encoded_goals:
        log(f"  {label}: prompt-toks={len(gt)} gold-cert-toks={len(ct)}")

    torch.manual_seed(42)
    random.seed(42)

    model = MicroGPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    log(f"num params: {sum(p.numel() for p in model.parameters())}")
    # Allow TF32 on the matmul path (3090 supports it).  Big win on
    # launch-bound autoregressive decode at small B.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # torch.compile: with mode="default" the inductor backend fuses the
    # block ops (rmsnorm + linear + relu + linear) into a smaller number
    # of kernels, cutting launch overhead.  This is the biggest single
    # lever for GPU utilisation at this model size since each generation
    # step is otherwise launching ~30 tiny kernels.  Disable with
    # HOL_COMPILE=0 if a backend issue arises.
    if torch.cuda.is_available() and int(os.environ.get("HOL_COMPILE", "1")):
        try:
            model = torch.compile(model, mode="default", dynamic=True)
            log("torch.compile: enabled (mode=default, dynamic=True)")
        except Exception as e:
            log(f"torch.compile failed: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.85, 0.99), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: max(0.0, 1 - s / max(1, NUM_STEPS)))

    baselines = {label: 0.0 for (label, _, _, _) in encoded_goals}
    start_step = 0
    if os.path.exists(CKPT_PATH):
        start_step, baselines = load_ckpt(CKPT_PATH, model, optimizer, device)
        for _ in range(start_step):
            scheduler.step()

    # Use all (or nearly all) CPU cores for the verifier subprocess pool.
    # Cap at the global rollouts/step since more verifiers than rollouts is
    # wasted, and at 48 to avoid pathological process spawn cost.
    n_verifiers = max(1, min(48, os.cpu_count() or 4, N_ROLLOUTS_PER_STEP))
    vpool = VerifierPool(n_verifiers, VERIFIER_BIN) if USE_KERNEL_VERIFY else None
    log(f"verifier pool: {n_verifiers} subprocesses" if vpool else "verifier: python fallback")

    if WARMUP_STEPS > 0 and start_step == 0:
        run_warmup(model, optimizer, device)

    threshold = (REWARD_CORRECT + REWARD_WRONG_CONCL) / 2.0

    try:
        t_start = time.time()
        last_log_t = t_start
        for step in range(start_step, NUM_STEPS):
            label, goal, goal_toks, _ = encoded_goals[step % len(encoded_goals)]
            prompt_toks = [BOS] + list(goal_toks) + [BOS]

            model.train()
            logp_sum, ent_sum, gens = rollout(
                model, prompt_toks, N_ROLLOUTS_PER_STEP, device)

            if vpool is not None:
                codes = vpool.score_many([(g, goal_toks) for g in gens])
                rewards = [remap_reward(c) for c in codes]
            else:
                rewards = [V_python(goal, g) for g in gens]
            rewards_t = torch.tensor(rewards, dtype=torch.float, device=device)

            b = baselines[label]
            adv = (rewards_t - b).detach() / max(1.0, abs(REWARD_CORRECT))
            eff_beta = torch.where(rewards_t < threshold,
                                   torch.full_like(rewards_t, ENTROPY_BETA),
                                   torch.zeros_like(rewards_t))
            loss = -(adv * logp_sum).mean() + -(eff_beta * ent_sum).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            baselines[label] = 0.9 * b + 0.1 * float(rewards_t.mean().item())

            if (step + 1) % LOG_EVERY == 0:
                now = time.time()
                rate = LOG_EVERY / max(1e-6, (now - last_log_t))
                last_log_t = now
                r_min = float(rewards_t.min().item())
                r_max = float(rewards_t.max().item())
                r_avg = float(rewards_t.mean().item())
                best_gen_len = max(len(g) for g in gens) if gens else 0
                avg_b = sum(baselines.values()) / len(baselines)
                log(f"step {step+1:5d}/{NUM_STEPS} | {label:14s} | "
                    f"R avg {r_avg:+9.2f} (min {r_min:+9.1f}, max {r_max:+9.1f}) | "
                    f"best-gen-len {best_gen_len:3d} | avg-b {avg_b:+9.2f} | "
                    f"{rate:.2f} steps/s")

            if (step + 1) % CKPT_EVERY == 0:
                save_ckpt(CKPT_PATH, model, optimizer, baselines, step + 1)

        log("=== Inference (greedy) ===")
        greedy_inference(model, device)
        save_ckpt(CKPT_PATH, model, optimizer, baselines, NUM_STEPS)
        log("=== run complete ===")
        if log_f is not None:
            log_f.close()
    finally:
        if vpool is not None:
            vpool.close()

if __name__ == "__main__":
    main()

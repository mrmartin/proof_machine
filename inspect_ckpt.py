"""Load a trained microgpt_with_RL_hol.py checkpoint and dump per-prompt
generation samples (greedy + a few temperature samples).  Useful for
post-mortem inspection.

Updated for the PyTorch trainer: reads `torch.save` checkpoints containing
state_dict + optimizer + baselines + config.  Refuses to load old
scalar-Value pickles (R1-R5)."""

import os
import sys
import math
import random
import argparse

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tokenizer", "python"))
import hol_tokenizer as T

# Reuse the model from the trainer to avoid drift.
sys.path.insert(0, os.path.dirname(__file__))
from microgpt_with_RL_hol import MicroGPT, V_python  # noqa: E402

CKPT = os.environ.get("HOL_CKPT_PATH",
                      os.path.join(os.path.dirname(__file__), "hol_rl_ckpt.pt"))
NSAMPLES = int(os.environ.get("HOL_NSAMPLES", "5"))
TEMP = float(os.environ.get("HOL_TEMP", "0.7"))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(0)
torch.manual_seed(0)

try:
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    if not isinstance(ckpt, dict) or 'model' not in ckpt:
        raise ValueError("missing 'model' key — probably an old scalar-Value pickle")
except Exception as e:
    print(f"ERROR: could not load {CKPT}: {e}", file=sys.stderr)
    print("Old R1-R5 hol_rl_*.pkl files are not loadable by the PyTorch trainer.",
          file=sys.stderr)
    sys.exit(1)

cfg = ckpt['config']
print(f"loaded {CKPT}")
print(f"  step={ckpt['step']}  config={cfg}")
print(f"  baselines={ckpt['baselines']}")

model = MicroGPT(cfg['vocab_size'], cfg['n_embd'], cfg['n_head'],
                 cfg['n_layer'], cfg['block_size']).to(DEVICE)
model.load_state_dict(ckpt['model'])
model.eval()

max_gen_tokens = int(os.environ.get("HOL_MAX_GEN", "80"))
block_size = cfg['block_size']
vocab_size = cfg['vocab_size']

NAT, BOOL = T.nat_ty(), T.bool_ty()
SEEDS = [
    ("refl_x",     T.mk_eq(T.mk_var("x", NAT), T.mk_var("x", NAT))),
    ("refl_y",     T.mk_eq(T.mk_var("y", NAT), T.mk_var("y", NAT))),
    ("refl_n",     T.mk_eq(T.mk_var("n", NAT), T.mk_var("n", NAT))),
    ("refl_p_bool",T.mk_eq(T.mk_var("p", BOOL), T.mk_var("p", BOOL))),
    ("refl_q_bool",T.mk_eq(T.mk_var("q", BOOL), T.mk_var("q", BOOL))),
    ("imp_p",      T.mk_imp(T.mk_var("p", BOOL), T.mk_var("p", BOOL))),
    ("imp_q",      T.mk_imp(T.mk_var("q", BOOL), T.mk_var("q", BOOL))),
]

def render(toks):
    return " ".join(T.tok_str(t) for t in toks)

@torch.no_grad()
def rollout(goal_toks, greedy=False, temp=1.0):
    BOS = T.BOS
    prompt = [BOS] + list(goal_toks) + [BOS]
    prompt_ids = torch.tensor(prompt, dtype=torch.long, device=DEVICE).unsqueeze(0)
    P = prompt_ids.shape[1]
    if P >= block_size:
        return []
    logits_last, kv = model(prompt_ids, kv_caches=None, start_pos=0)
    state = T.initial_state()
    out = []
    max_iters = min(max_gen_tokens, block_size - P)
    for t in range(max_iters):
        row = T.valid_next_mask(state)
        mask = torch.as_tensor([row], dtype=torch.bool, device=DEVICE)
        masked = logits_last.masked_fill(~mask, float('-inf'))
        if greedy:
            sampled = int(masked.argmax(-1).item())
        else:
            logits = masked / temp
            probs = F.softmax(logits, dim=-1)
            sampled = int(torch.multinomial(probs, num_samples=1).item())
        ns = T.step(state, sampled)
        if ns is None: break
        state = ns
        out.append(sampled)
        if T.is_accepting(ns): break
        if t + 1 >= max_iters: break
        ids = torch.tensor([[sampled]], dtype=torch.long, device=DEVICE)
        logits_last, kv = model(ids, kv_caches=kv, start_pos=P + t)
    return out

for label, goal in SEEDS:
    goal_toks, _ = T.encode_term_only(goal)
    print(f"\n=== {label}  goal={render(goal_toks)} ===")
    g = rollout(goal_toks, greedy=True)
    r = V_python(goal, g)
    print(f"  greedy  R={r:+8.1f}  ({len(g)} toks): {render(g)}")
    for i in range(NSAMPLES):
        g = rollout(goal_toks, greedy=False, temp=TEMP)
        r = V_python(goal, g)
        print(f"  T={TEMP}   R={r:+8.1f}  ({len(g)} toks): {render(g)}")

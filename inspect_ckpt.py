"""Load a trained microgpt_with_RL_hol.py checkpoint and dump per-prompt
generation samples (greedy + a few temperature samples).  Useful for
post-mortem inspection."""

import os, sys, pickle, math, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tokenizer", "python"))
import hol_tokenizer as T

CKPT = os.environ.get("HOL_CKPT_PATH",
                     os.path.join(os.path.dirname(__file__), "hol_rl_ckpt.pkl"))
NSAMPLES = int(os.environ.get("HOL_NSAMPLES", "5"))
TEMP = float(os.environ.get("HOL_TEMP", "0.7"))

# --- Identical model architecture as the training script ---------------------
random.seed(0)
n_layer, n_embd, n_head = 1, 12, 4
head_dim = n_embd // n_head
block_size = 80
max_gen_tokens = 64
vocab_size = T.VOCAB_SIZE

class Value:
    __slots__ = ('data',)
    def __init__(self, data): self.data = data

def matrix(nout, nin):
    return [[Value(0.0) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd),
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row]

with open(CKPT, 'rb') as f:
    ckpt = pickle.load(f)
for p, d in zip(params, ckpt['params_data']):
    p.data = d
print(f"loaded ckpt: step={ckpt['step']}, params={len(params)}")
print(f"baselines: {ckpt['baselines']}")

# --- Forward (data-only, no autograd) ---------------------------------------
def linear(x, w):
    return [sum(wi.data * xi for wi, xi in zip(wo, x)) for wo in w]
def softmax(logits):
    mv = max(logits)
    exps = [math.exp(v - mv) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = [w.data for w in state_dict['wte'][token_id]]
    pos_emb = [w.data for w in state_dict['wpe'][pos_id]]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)
    for li in range(n_layer):
        x_residual = x; x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k); values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                           for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                        for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x; x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [max(0, xi) for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
    return linear(x, state_dict['lm_head'])

# --- Seeds (same as training) -----------------------------------------------
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

def _infer_header(toks):
    hdr = T.PoolHeader()
    seen_v = set(); seen_n = set(); seen_tc = set(); seen_tv = set()
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

def V(goal, gen_tokens):
    s = T.initial_state()
    for t in gen_tokens:
        ns = T.step(s, t)
        if ns is None: return -1
        s = ns
    if not T.is_accepting(s): return -1
    try:
        hdr = _infer_header(gen_tokens)
        cert = T.decode_cert(hdr, gen_tokens)
    except Exception:
        return +1
    if T.alpha_eq(cert.concl, goal): return +100
    return +1

def render(toks):
    return " ".join(T.tok_str(t) for t in toks)

def rollout(goal_toks, greedy=False, temp=1.0):
    BOS = T.BOS
    prompt = [BOS] + list(goal_toks) + [BOS]
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    for pid, tok in enumerate(prompt):
        _ = gpt(tok, pid, keys, values)
    state = T.initial_state()
    out = []
    last = prompt[-1]
    pos = len(prompt)
    gen = 0
    while pos < block_size and gen < max_gen_tokens:
        logits = gpt(last, pos, keys, values)
        mask = T.valid_next_mask(state)
        valid = [(i, logits[i]) for i, ok in enumerate(mask) if ok]
        if not valid: break
        if greedy:
            sampled = max(valid, key=lambda kv: kv[1])[0]
        else:
            vals = [l / temp for _, l in valid]
            mv = max(vals)
            exps = [math.exp(v - mv) for v in vals]
            s = sum(exps); probs = [e / s for e in exps]
            k = random.choices(range(len(valid)), weights=probs)[0]
            sampled = valid[k][0]
        out.append(sampled)
        ns = T.step(state, sampled)
        if ns is None: break
        state = ns
        last = sampled
        pos += 1; gen += 1
        if T.is_accepting(state): break
    return out

for label, goal in SEEDS:
    goal_toks, _ = T.encode_term_only(goal)
    print(f"\n=== {label}  goal={render(goal_toks)} ===")
    g = rollout(goal_toks, greedy=True)
    r = V(goal, g)
    print(f"  greedy  R={r:+d}  ({len(g)} toks): {render(g)}")
    for i in range(NSAMPLES):
        g = rollout(goal_toks, greedy=False, temp=TEMP)
        r = V(goal, g)
        print(f"  T={TEMP}   R={r:+d}  ({len(g)} toks): {render(g)}")

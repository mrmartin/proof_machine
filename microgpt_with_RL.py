"""
microgpt_rl.py
The most atomic way to train a GPT with REINFORCE against a fixed validator V.

Setting:
  - A set of seed prompts.
  - A validator V(prompt, completion) -> {-1, +1, +100}
      false             = -1
      true              = +1
      true_and_helpful  = +100
  - We want pi_theta(completion | prompt) to put mass on high-V completions.

Algorithm: vanilla REINFORCE with a per-prompt EMA baseline.
  loss = -(R - b_prompt) * sum_t log pi(a_t | s_<t)
where {a_t} are the GENERATED tokens only (prompt tokens are conditioning).

@karpathy original + RL conversion
"""

import os
import math
import random
random.seed(42)

# -----------------------------------------------------------------------------
# The validator V. Black-box, fast, returns one of three discrete scores.
# Swap this for your real validator. Toy example below: prompt is a vowel,
# completion should be all-consonants; +100 if also exactly 3 chars long.
VOWELS = set('aeiou')
def V(prompt, completion):
    if any(c in VOWELS for c in completion):
        return -1                    # false: emitted a vowel
    if len(completion) == 3:
        return 100                   # true_and_helpful: consonants AND length 3
    if len(completion) > 0:
        return 1                     # true: consonants but wrong length
    return -1                        # empty completion: not helpful

seed_prompts = list('aeiou')
print(f"num prompts: {len(seed_prompts)}")

# -----------------------------------------------------------------------------
# Tokenizer: fix the vocab to a-z so V's vowel check lines up with the model.
uchars = sorted(set('abcdefghijklmnopqrstuvwxyz'))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# -----------------------------------------------------------------------------
# Autograd: identical to microgpt.py
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    def __init__(self, data, children=(), local_grads=()):
        self.data = data; self.grad = 0
        self._children = children; self._local_grads = local_grads
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    def backward(self):
        topo = []; visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children: build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# -----------------------------------------------------------------------------
# Model: identical to microgpt.py
n_layer = 1; n_embd = 16; block_size = 16; n_head = 4; head_dim = n_embd // n_head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
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
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x; x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
    logits = linear(x, state_dict['lm_head'])
    return logits

# -----------------------------------------------------------------------------
# Adam state: identical
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params); v_buf = [0.0] * len(params)

# -----------------------------------------------------------------------------
# Per-prompt EMA baseline. With reward in {-1, +1, +100} the expected return
# per prompt can swing enormously between prompts as training progresses, so
# a global baseline would be a poor variance reducer.
baselines = {p: 0.0 for p in seed_prompts}
baseline_ema = 0.9

# -----------------------------------------------------------------------------
# Training loop
num_steps = 2000
for step in range(num_steps):

    # 1) Pick a seed prompt.
    prompt = seed_prompts[step % len(seed_prompts)]
    prompt_tokens = [BOS] + [uchars.index(ch) for ch in prompt]

    # 2) Forward through the prompt to populate the KV cache.
    # These tokens are conditioning, not actions: we discard their logits.
    # Gradients still flow through their K/V projections via attention from
    # later (generated) positions, which is what we want.
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    for pos_id, tok in enumerate(prompt_tokens):
        _ = gpt(tok, pos_id, keys, values)

    # 3) Roll out a completion, recording log pi(a_t | s_<t) of each sample.
    log_probs = []   # list[Value]
    generated = []   # list[int]
    last_token = prompt_tokens[-1]
    pos = len(prompt_tokens)
    while pos < block_size:
        logits = gpt(last_token, pos, keys, values)
        probs = softmax(logits)
        sampled = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        log_probs.append(probs[sampled].log())
        if sampled == BOS:
            break
        generated.append(sampled)
        last_token = sampled
        pos += 1

    completion = ''.join(uchars[t] for t in generated)

    # 4) Score and form the advantage against this prompt's baseline.
    reward = V(prompt, completion)
    b = baselines[prompt]
    advantage = reward - b
    baselines[prompt] = baseline_ema * b + (1 - baseline_ema) * reward

    # 5) REINFORCE: minimize  -advantage * sum_t log pi(a_t | s_<t).
    # `advantage` is a plain float, treated as a constant multiplier on the
    # gradient — this is the textbook REINFORCE estimator.
    if len(log_probs) > 0:
        loss = -advantage * sum(log_probs)
        loss.backward()

        # 6) Adam step (identical to microgpt.py).
        lr_t = learning_rate * (1 - step / num_steps)
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v_buf[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = 0
    else:
        # Sampled BOS immediately — no actions to credit. Zero stray grads.
        for p in params: p.grad = 0

    if (step + 1) % 50 == 0:
        avg_b = sum(baselines.values()) / len(baselines)
        print(f"step {step+1:4d}/{num_steps} | '{prompt}' -> '{completion}' | R {reward:+4d} | avg baseline {avg_b:+7.2f}")

# -----------------------------------------------------------------------------
# Inference: sample one completion per prompt and show the validator's verdict.
print("\n--- inference (one sample per seed prompt) ---")
temperature = 0.5
for prompt in seed_prompts:
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    prompt_tokens = [BOS] + [uchars.index(ch) for ch in prompt]
    for pos_id, tok in enumerate(prompt_tokens):
        _ = gpt(tok, pos_id, keys, values)
    last_token = prompt_tokens[-1]
    out = []
    pos = len(prompt_tokens)
    while pos < block_size:
        logits = gpt(last_token, pos, keys, values)
        probs = softmax([l / temperature for l in logits])
        sampled = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if sampled == BOS: break
        out.append(uchars[sampled])
        last_token = sampled
        pos += 1
    completion = ''.join(out)
    print(f"  '{prompt}' -> '{completion}'  V={V(prompt, completion):+d}")

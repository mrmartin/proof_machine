"""tree_search_infer.py — M1 rule-level kernel-pruned tree search.

The policy proposes; the kernel disposes.

At every step boundary we:
  1) Generate the structural prelude "(step k (rule" deterministically
     under the PDA mask.
  2) At the rule-choice position, branch into the top-k rule choices.
  3) For each branch, greedily complete the step body (witness +
     premises) under the PDA mask.
  4) Send the new prefix to the kernel via verify_prefix.  If the kernel
     rejects, drop the branch.  If it accepts, push (logp, prefix,
     theorem_table, grammar_state) onto the frontier.

A node is terminal when its newest derived theorem's conclusion is
alpha-equivalent to the goal — checked by wrapping the prefix with a
declared (concl <goal>) block and running the legacy verifier.  Both
sides parse against their own pool, so alpha-equivalence at the kernel
level handles the slot mismatch.

Per-goal budget is measured in kernel calls (verify_prefix + legacy
verify).  Each forward pass through the model is essentially free
compared to the kernel walk; we still cap rule-choice forward passes
as a secondary brake.

Used by:
  - eval_ood.py runs 1A (flat baseline) / 1B (best-first) on the 23-seed
    test set.
"""
from __future__ import annotations

import heapq
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tokenizer", "python"))
sys.path.insert(0, HERE)

import torch                                # noqa: E402
import torch.nn.functional as F             # noqa: E402

import hol_tokenizer as T                   # noqa: E402
from kernel_client import (                 # noqa: E402
    KernelClient, PrefixResult, ThmRecord,
)
from microgpt_with_RL_hol_gpu import (      # noqa: E402
    HOLGPT, N_LAYER, N_HEAD, N_EMBD,
)


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------
RULE_MASK_INDICES = set(range(T.RULE_FIRST, T.RULE_LAST + 1))

# Rule → required witness keyword token (None = no witness body, i.e.
# (witness) is empty).  This is a *syntactic* constraint imposed by
# Cert.apply_step's pattern match in kernel/cert.ml — the same rule can
# only consume one witness ADT shape, and that shape is keyed by the
# witness's leading keyword.  Encoding this as a hard constraint in
# the search (rather than letting the policy guess) eliminates a whole
# class of trivial failures.
_RULE_TO_WITNESS_KW = {
    "REFL":    T.KW_TERM,
    "BETA":    T.KW_TERM,
    "ASSUME":  T.KW_TERM,
    "SPEC":    T.KW_TERM,
    "DISCH":   T.KW_TERM,
    "ABS":     T.KW_VAR,
    "GEN":     T.KW_VAR,
    "CHOOSE":  T.KW_VAR,
    "INST":    T.KW_INST,
    "INST_TYPE": T.KW_INSTTYPE,
    "EXISTS":  T.KW_B_AND_W,
    "AXIOM":   T.KW_AXIOM,
    # Rules below take W_none → empty witness body.
    "TRANS":               None,
    "MK_COMB":             None,
    "EQ_MP":               None,
    "DEDUCT_ANTISYM_RULE": None,
    "CONJ":                None,
    "CONJUNCT1":           None,
    "CONJUNCT2":           None,
    "MP":                  None,
    "ETA_AX":              None,
    "SELECT_AX":           None,
    "EM_AX":               None,
}
RULE_TO_WITNESS_KW: dict = {
    T.RULE_TOK[name]: kw for name, kw in _RULE_TO_WITNESS_KW.items()
}


def is_rule_choice_position(mask: Sequence[bool]) -> bool:
    """The unique PDA state where the only valid next tokens are rule
    names (after `LPAREN KW_RULE`).  Detected by exact set equality."""
    allowed = {i for i, m in enumerate(mask) if m}
    return allowed == RULE_MASK_INDICES


_WITNESS_KW_TOKENS = {T.KW_TERM, T.KW_TYPE, T.KW_VAR, T.KW_INST,
                       T.KW_INSTTYPE, T.KW_AXIOM, T.KW_B_AND_W}


def is_witness_kw_position(mask: Sequence[bool]) -> bool:
    """The unique PDA state where the next token must be one of the
    seven witness keywords (after `(witness (`).  Detected by checking
    that all allowed tokens are in _WITNESS_KW_TOKENS."""
    allowed = {i for i, m in enumerate(mask) if m}
    return allowed.issubset(_WITNESS_KW_TOKENS) and len(allowed) > 1


# ---------------------------------------------------------------------------
# Search nodes
# ---------------------------------------------------------------------------
@dataclass(order=False)
class Node:
    step_toks: List[int]
    n_steps: int
    cum_logp: float
    grammar_state: object
    theorems: List[ThmRecord]

    @property
    def avg_logp(self) -> float:
        if self.n_steps == 0:
            return 0.0
        return self.cum_logp / self.n_steps

    def priority(self, alpha: float = 1.0) -> float:
        # Higher score = better.  Length-normalised log-probability.
        if self.n_steps == 0:
            return 0.0
        return self.cum_logp / (self.n_steps ** alpha)


# ---------------------------------------------------------------------------
# Cert-token assembly: build a verifier-acceptable cert from a list of
# step blocks (raw tokens emitted by the model) plus a declared concl.
# ---------------------------------------------------------------------------
def wrap_cert(step_toks: Sequence[int], concl_toks: Sequence[int]) -> List[int]:
    """Returns `( cert <step_blocks> ( concl " <concl_toks> " ) )`."""
    return ([T.LPAREN, T.KW_CERT]
            + list(step_toks)
            + [T.LPAREN, T.KW_CONCL, T.QUOTE]
            + list(concl_toks)
            + [T.QUOTE, T.RPAREN, T.RPAREN])


# ---------------------------------------------------------------------------
# Decoding: generate one full step block starting from a given PDA state.
# - greedy=True: argmax at every position
# - rule_override: if set, force this exact rule token at the rule-choice
#   position (used when the search has already chosen a top-k rule).
# Returns (step_toks, log_prob_at_rule_choice, final_grammar_state) or
# None if decoding can't complete the step within max_step_toks.
# ---------------------------------------------------------------------------
def _emit_one_step(model: HOLGPT,
                   prompt: List[int],
                   step_toks_so_far: List[int],
                   grammar_state,
                   device: torch.device,
                   max_step_toks: int = 80,
                   temperature: float = 0.7,
                   rule_override: Optional[int] = None,
                   greedy_after_rule: bool = True,
                   ) -> Optional[Tuple[List[int], float, object]]:
    """Emit tokens until exactly one step block has been completed,
    i.e. until the PDA returns to a state immediately after the closing
    RPAREN of `(step ...)` (one step deeper than initial-cert-state).

    Detection: after each token, if the PDA mask now includes LPAREN
    AND the next-step-id (which is INT_<n_steps + 1>) is valid, we've
    just closed a step.  Simpler heuristic: track parenthesis depth.

    Returns the step's tokens (not including pre-existing context) and
    the log-prob at the rule-choice position (the rest is greedy /
    forced).
    """
    state = grammar_state
    out: List[int] = []
    rule_logp = 0.0
    rule_chosen = False
    # If a rule was forced, remember which witness keyword must come
    # next so we can pin it at the witness-kw position too.
    forced_witness_kw: Optional[int] = None
    witness_kw_pinned = (rule_override is None)
    witness_body_pinned = (rule_override is None)
    last_tok: Optional[int] = None
    # Parenthesis depth from start of this step.  A step block is
    # `( step <id> ( rule R ) ( witness ... ) ( premises ... ) )` —
    # opens with LPAREN, closes when we return to depth 0.
    depth = 0
    depth_started = False
    cur_seq = list(prompt) + list(step_toks_so_far)
    block_size = getattr(model, "block_size", 320)

    for _step_i in range(max_step_toks):
        if len(cur_seq) >= block_size:
            return None  # exceeded position embedding range
        cur = torch.tensor([cur_seq], dtype=torch.long, device=device)
        with torch.no_grad():
            logits_all = model(cur)
        logits = logits_all[0, -1]
        mask = T.valid_next_mask(state)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=device)

        if not mask_t.any():
            return None  # PDA stuck (shouldn't happen in practice)

        # Rule-choice position?
        if rule_override is not None and is_rule_choice_position(mask) and not rule_chosen:
            tok = rule_override
            if not mask[tok]:
                return None
            log_probs = F.log_softmax(
                logits.masked_fill(~mask_t, float("-inf")), dim=-1)
            rule_logp = float(log_probs[tok].item())
            rule_chosen = True
            forced_witness_kw = RULE_TO_WITNESS_KW.get(rule_override)
        elif (rule_chosen and not witness_body_pinned
              and last_tok == T.KW_WITNESS
              and mask[T.LPAREN] and mask[T.RPAREN]):
            # Post-`( witness` choice between empty body (RPAREN) and
            # opening a body (LPAREN).  W_none rules need empty body;
            # all others need an opened body.
            witness_body_pinned = True
            if forced_witness_kw is None:
                tok = T.RPAREN  # empty witness for W_none rules
            else:
                tok = T.LPAREN
            if not mask[tok]:
                return None
        elif is_rule_choice_position(mask) and not rule_chosen and rule_override is None:
            log_probs = F.log_softmax(
                logits.masked_fill(~mask_t, float("-inf")) / max(temperature, 1e-6),
                dim=-1)
            probs = log_probs.exp()
            tok = int(torch.multinomial(probs, num_samples=1).item())
            rule_logp = float(log_probs[tok].item())
            rule_chosen = True
            forced_witness_kw = RULE_TO_WITNESS_KW.get(tok)
        elif (rule_chosen and not witness_kw_pinned
              and forced_witness_kw is not None
              and is_witness_kw_position(mask)):
            # Pin the witness keyword to match the chosen rule's expected
            # witness shape.  This is the syntactic constraint imposed
            # by apply_step in kernel/cert.ml.
            tok = forced_witness_kw
            if not mask[tok]:
                return None
            witness_kw_pinned = True
        elif (rule_chosen and not witness_kw_pinned
              and forced_witness_kw is None
              and is_witness_kw_position(mask)):
            # Rule expects empty witness — but PDA still requires us to
            # pick a keyword.  Pick the first allowed keyword and let
            # the rest of the witness body be sampled (kernel will
            # accept any valid body since W_none-rules ignore witness
            # contents — except they fail in apply_step if witness !=
            # W_none).  In practice we should NOT reach this position
            # for W_none rules: the empty (witness ) block has no
            # keyword.  Fall through to general sampling.
            witness_kw_pinned = True
            if greedy_after_rule:
                masked = logits.masked_fill(~mask_t, float("-inf"))
                tok = int(masked.argmax().item())
            else:
                log_probs = F.log_softmax(
                    logits.masked_fill(~mask_t, float("-inf")) / max(temperature, 1e-6),
                    dim=-1)
                probs = log_probs.exp()
                tok = int(torch.multinomial(probs, num_samples=1).item())
        else:
            # Greedy on allowed tokens (deterministic step body).
            if greedy_after_rule or not rule_chosen:
                masked = logits.masked_fill(~mask_t, float("-inf"))
                tok = int(masked.argmax().item())
            else:
                log_probs = F.log_softmax(
                    logits.masked_fill(~mask_t, float("-inf")) / max(temperature, 1e-6),
                    dim=-1)
                probs = log_probs.exp()
                tok = int(torch.multinomial(probs, num_samples=1).item())

        # Advance PDA.
        nstate = T.step(state, tok)
        if nstate is None:
            return None
        state = nstate

        if tok == T.LPAREN:
            depth += 1
            depth_started = True
        elif tok == T.RPAREN:
            depth -= 1

        out.append(tok)
        cur_seq.append(tok)
        last_tok = tok

        # End of step block reached when we close back to depth 0
        # after having opened at least once.
        if depth_started and depth == 0:
            return (out, rule_logp, state)

    return None  # Failed to close the step within budget.


# ---------------------------------------------------------------------------
# Get rule-choice logits at the start of the next step (used by 1B for
# top-k expansion before committing).  Returns (rule_tokens_sorted,
# log_probs_sorted, prelude_tokens, prelude_state).
# ---------------------------------------------------------------------------
def _rule_logits_at_next_step(model: HOLGPT,
                              prompt: List[int],
                              step_toks_so_far: List[int],
                              grammar_state,
                              device: torch.device,
                              max_prelude: int = 8,
                              ) -> Optional[Tuple[List[int], List[float], List[int], object]]:
    """Greedy-decode the deterministic prelude of the next step (LPAREN,
    KW_STEP, INT_<id>, LPAREN, KW_RULE) until the rule-choice position,
    then return the masked log-probabilities over rule tokens."""
    state = grammar_state
    prelude: List[int] = []
    cur_seq = list(prompt) + list(step_toks_so_far)
    block_size = getattr(model, "block_size", 320)

    for _ in range(max_prelude):
        if len(cur_seq) >= block_size:
            return None
        cur = torch.tensor([cur_seq], dtype=torch.long, device=device)
        with torch.no_grad():
            logits_all = model(cur)
        logits = logits_all[0, -1]
        mask = T.valid_next_mask(state)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
        if is_rule_choice_position(mask):
            log_probs = F.log_softmax(
                logits.masked_fill(~mask_t, float("-inf")), dim=-1)
            rule_tokens = list(range(T.RULE_FIRST, T.RULE_LAST + 1))
            lps = [float(log_probs[t].item()) for t in rule_tokens]
            order = sorted(range(len(rule_tokens)),
                           key=lambda i: -lps[i])
            sorted_tokens = [rule_tokens[i] for i in order]
            sorted_lps = [lps[i] for i in order]
            return (sorted_tokens, sorted_lps, prelude, state)
        # Otherwise: deterministic prelude — argmax under PDA mask.
        if not mask_t.any():
            return None
        masked = logits.masked_fill(~mask_t, float("-inf"))
        tok = int(masked.argmax().item())
        nstate = T.step(state, tok)
        if nstate is None:
            return None
        prelude.append(tok)
        cur_seq.append(tok)
        state = nstate
    return None


# ---------------------------------------------------------------------------
# Search routines
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    solved: bool
    found_cert: Optional[List[int]] = None
    kernel_calls: int = 0
    forward_passes: int = 0
    depth_reached: int = 0
    elapsed: float = 0.0
    rule_shape: Tuple[str, ...] = ()
    novel_shape: bool = False


def _certify_terminal(client: KernelClient,
                      step_toks: Sequence[int],
                      concl_toks: Sequence[int],
                      goal_toks: Sequence[int],
                      cert_names: Sequence[str],
                      goal_names: Sequence[str]) -> bool:
    """Wrap a prefix into a full cert with the derived concl and check
    legacy verify accepts.  Both pools are reinferred by the verifier."""
    cert = wrap_cert(step_toks, concl_toks)
    v = client.verify(cert, list(goal_toks),
                      cert_names=list(cert_names),
                      goal_names=list(goal_names))
    return v == 100


def _cert_rule_shape(cert_toks: Sequence[int]) -> Tuple[str, ...]:
    """Extract rule sequence from raw cert tokens."""
    from microgpt_expit import cert_rule_shape  # reuse existing impl
    return cert_rule_shape(cert_toks)


def extract_subterms(goal_toks: Sequence[int]) -> List[List[int]]:
    """Return all balanced-parenthesis sub-token-sequences of the goal.
    Used as a structural prior on W_term witnesses: ASSUME/DISCH and
    similar rules typically take a subterm of the goal.

    The token-level definition: a subterm is either (a) an atomic
    token (single VAR/NAME/INT) followed by `: <type>` (for vars)
    or just the const/int (for constants), or (b) a parenthesised
    expression `( ... )` with balanced parens.  We don't decode all
    the way to the AST — token-level subterms are enough for the
    search to test as witness candidates.
    """
    out: List[List[int]] = []
    # All balanced-paren subsequences.
    starts = []
    for i, t in enumerate(goal_toks):
        if t == T.LPAREN:
            starts.append(i)
        elif t == T.RPAREN and starts:
            s = starts.pop()
            out.append(list(goal_toks[s:i + 1]))
    # Atomic vars with their types: `VAR_k : <type-toks>`.
    i = 0
    while i < len(goal_toks):
        t = goal_toks[i]
        if T.VAR_FIRST <= t <= T.VAR_LAST or T.NAME_FIRST <= t <= T.NAME_LAST:
            # Find the next non-type-related token to delimit.  We greedily
            # consume `: <type>` if present.
            j = i + 1
            if j < len(goal_toks) and goal_toks[j] == T.COLON:
                j += 1
                # Type can be a single token (bool/nat/tycon) or
                # a parenthesised expression.
                if j < len(goal_toks) and goal_toks[j] == T.LPAREN:
                    depth = 1
                    j += 1
                    while j < len(goal_toks) and depth > 0:
                        if goal_toks[j] == T.LPAREN: depth += 1
                        elif goal_toks[j] == T.RPAREN: depth -= 1
                        j += 1
                else:
                    j += 1
                out.append(list(goal_toks[i:j]))
        i += 1
    # Deduplicate while preserving order.
    seen = set()
    uniq: List[List[int]] = []
    for s in out:
        key = tuple(s)
        if key in seen: continue
        seen.add(key)
        uniq.append(s)
    return uniq


def search_best_first(model: HOLGPT,
                       client: KernelClient,
                       prompt: List[int],
                       goal_toks: List[int],
                       cert_names: List[str],
                       goal_names: List[str],
                       device: torch.device,
                       k_outer: int = 5,
                       b_inner: int = 4,
                       inner_temp: float = 0.9,
                       max_depth: int = 12,
                       max_kernel_calls: int = 1000,
                       alpha: float = 1.0,
                       uniform_mix: float = 0.0,
                       known_corpus_shapes: Optional[set] = None,
                       ) -> SearchResult:
    """Best-first rule-level tree search.  Returns SearchResult."""
    t0 = time.time()
    initial_state = T.initial_state()
    # Step 0: open the cert with `( KW_cert` — required by the PDA before
    # any (step ...) can appear.
    s0 = T.step(initial_state, T.LPAREN)
    if s0 is None:
        return SearchResult(solved=False)
    s0 = T.step(s0, T.KW_CERT)
    if s0 is None:
        return SearchResult(solved=False)

    root = Node(step_toks=[T.LPAREN, T.KW_CERT],
                n_steps=0,
                cum_logp=0.0,
                grammar_state=s0,
                theorems=[])

    # Use a heap of (-priority, counter, Node) tuples.  The counter
    # breaks ties deterministically.
    counter = 0
    frontier: List[Tuple[float, int, Node]] = []
    heapq.heappush(frontier, (0.0, counter, root))

    kernel_calls = 0
    forward_passes = 0
    depth_reached = 0
    novel_corpus_shapes = known_corpus_shapes or set()

    while frontier and kernel_calls < max_kernel_calls:
        neg_prio, _, node = heapq.heappop(frontier)
        if node.n_steps >= max_depth:
            continue
        depth_reached = max(depth_reached, node.n_steps)

        # Get rule-choice logits at the next step boundary.
        rule_info = _rule_logits_at_next_step(
            model, prompt, node.step_toks, node.grammar_state, device)
        forward_passes += 1
        if rule_info is None:
            continue
        sorted_rule_tokens, sorted_rule_lps, prelude_toks, prelude_state = rule_info

        # Apply uniform mixing if requested (1B-f ablation).
        if uniform_mix > 0:
            log_uniform = math.log(1.0 / len(sorted_rule_tokens))
            mixed = []
            for lp in sorted_rule_lps:
                p = (1 - uniform_mix) * math.exp(lp) + uniform_mix * math.exp(log_uniform)
                mixed.append(math.log(max(p, 1e-12)))
            order = sorted(range(len(mixed)), key=lambda i: -mixed[i])
            sorted_rule_tokens = [sorted_rule_tokens[i] for i in order]
            sorted_rule_lps = [mixed[i] for i in order]

        # Deduplicate candidate steps emitted from this node across all
        # (rule × inner-sample) attempts, so we don't kernel-verify the
        # same body twice when sampling is degenerate.
        seen_blocks = set()
        for rule_idx, (rule_tok, rule_lp) in enumerate(
                zip(sorted_rule_tokens, sorted_rule_lps)):
            if rule_idx >= k_outer:
                break
            if kernel_calls >= max_kernel_calls:
                break

            for inner_i in range(b_inner):
                if kernel_calls >= max_kernel_calls:
                    break

                # Emit one step with the rule forced; sample the body
                # (witness + premises) with temperature so different
                # witness shapes get explored.
                emit = _emit_one_step(
                    model, prompt, node.step_toks,
                    node.grammar_state, device,
                    rule_override=rule_tok,
                    temperature=inner_temp,
                    greedy_after_rule=(inner_i == 0))
                forward_passes += 1
                if emit is None:
                    continue
                step_block, _logp_at_rule, new_state = emit
                block_key = tuple(step_block)
                if block_key in seen_blocks:
                    continue
                seen_blocks.add(block_key)

                new_step_toks = node.step_toks + step_block
                new_n_steps = node.n_steps + 1

                # Verify the partial cert via the kernel.
                body = list(new_step_toks)[2:]  # drop leading ( KW_cert
                wrapped = wrap_cert(body, goal_toks)
                pres = client.verify_prefix(wrapped, new_n_steps,
                                             cert_names=cert_names)
                kernel_calls += 1
                if not pres.ok or not pres.theorems:
                    continue

                cum_lp = node.cum_logp + rule_lp
                child = Node(step_toks=new_step_toks,
                              n_steps=new_n_steps,
                              cum_logp=cum_lp,
                              grammar_state=new_state,
                              theorems=pres.theorems)

                last_thm = pres.theorems[-1]
                if _certify_terminal(client, body, last_thm.concl_toks,
                                      goal_toks, cert_names, goal_names):
                    kernel_calls += 1
                    full_cert = wrap_cert(body, last_thm.concl_toks)
                    shape = _cert_rule_shape(full_cert)
                    return SearchResult(
                        solved=True, found_cert=full_cert,
                        kernel_calls=kernel_calls,
                        forward_passes=forward_passes,
                        depth_reached=new_n_steps,
                        elapsed=time.time() - t0,
                        rule_shape=shape,
                        novel_shape=shape not in novel_corpus_shapes,
                    )
                kernel_calls += 1

                counter += 1
                heapq.heappush(frontier,
                                (-child.priority(alpha), counter, child))

    return SearchResult(
        solved=False, kernel_calls=kernel_calls,
        forward_passes=forward_passes,
        depth_reached=depth_reached,
        elapsed=time.time() - t0)


def search_flat_sample(model: HOLGPT,
                       client: KernelClient,
                       prompt: List[int],
                       goal_toks: List[int],
                       cert_names: List[str],
                       goal_names: List[str],
                       device: torch.device,
                       num_samples: int = 64,
                       temperature: float = 1.0,
                       max_depth: int = 12,
                       known_corpus_shapes: Optional[set] = None,
                       ) -> SearchResult:
    """Flat PDA-masked sampling control (1A).  Decodes complete certs
    one step at a time without any tree backtracking, just like the
    existing sample_rollouts path, then kernel-verifies each."""
    t0 = time.time()
    initial_state = T.initial_state()
    s0 = T.step(initial_state, T.LPAREN)
    if s0 is None:
        return SearchResult(solved=False)
    s0 = T.step(s0, T.KW_CERT)
    if s0 is None:
        return SearchResult(solved=False)

    kernel_calls = 0
    forward_passes = 0
    novel_corpus_shapes = known_corpus_shapes or set()

    for sample_i in range(num_samples):
        state = s0
        step_toks = [T.LPAREN, T.KW_CERT]
        cum_lp = 0.0
        n_steps = 0
        success = False
        last_concl: Optional[List[int]] = None

        for _ in range(max_depth):
            emit = _emit_one_step(
                model, prompt, step_toks, state, device,
                rule_override=None,  # sample the rule
                temperature=temperature,
                greedy_after_rule=False if temperature > 0 else True)
            forward_passes += 1
            if emit is None:
                break
            step_block, rule_lp, new_state = emit
            step_toks = step_toks + step_block
            n_steps += 1
            cum_lp += rule_lp
            state = new_state

            # Verify after every step (kernel-pruned even in flat).
            body = step_toks[2:]  # drop leading ( KW_cert
            wrapped = wrap_cert(body, goal_toks)
            pres = client.verify_prefix(wrapped, n_steps,
                                         cert_names=cert_names)
            kernel_calls += 1
            if not pres.ok or not pres.theorems:
                break
            last_thm = pres.theorems[-1]
            if _certify_terminal(client, body, last_thm.concl_toks,
                                  goal_toks, cert_names, goal_names):
                kernel_calls += 1
                last_concl = last_thm.concl_toks
                success = True
                break
            kernel_calls += 1

        if success:
            full_cert = wrap_cert(step_toks[2:], last_concl)
            shape = _cert_rule_shape(full_cert)
            return SearchResult(
                solved=True, found_cert=full_cert,
                kernel_calls=kernel_calls,
                forward_passes=forward_passes,
                depth_reached=n_steps,
                elapsed=time.time() - t0,
                rule_shape=shape,
                novel_shape=shape not in novel_corpus_shapes,
            )

    return SearchResult(
        solved=False, kernel_calls=kernel_calls,
        forward_passes=forward_passes,
        depth_reached=max_depth,
        elapsed=time.time() - t0)

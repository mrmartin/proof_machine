"""synth/freq_counter.py — position-aware rule frequency counter.

Streams the persistent buffer JSONL (`hol_expit_buffer.jsonl`) once at
generator init, extracts each cert's rule sequence, and tallies how
often each rule appears at each position.  The result drives the
inverse-frequency adaptive sampler in `synth/backward_gen.py`.

Kept out of `backward_gen.py` so it can be reused (e.g. by an offline
audit) and tested in isolation.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tokenizer", "python"))

import hol_tokenizer as T


def cert_rule_shape(cert_toks: Sequence[int]) -> Tuple[str, ...]:
    """Ordered tuple of rule names used in a cert.  Inlined here (not
    imported from microgpt_expit) to keep this module free of trainer
    dependencies — microgpt_expit's import chain pulls in PyTorch and
    opens log files."""
    rules = []
    n = len(cert_toks)
    for i in range(n - 1):
        if cert_toks[i] == T.KW_RULE:
            for j in range(i + 1, min(i + 4, n)):
                t = cert_toks[j]
                if T.RULE_FIRST <= t <= T.RULE_LAST:
                    rules.append(T.RULE_FROM_TOK[t])
                    break
    return tuple(rules)


@dataclass
class PositionRuleFreq:
    """counts[position][rule_name] -> int.  total_proofs is bookkeeping
    for diagnostics; the sampler only consults counts."""
    counts: Dict[int, Dict[str, int]] = field(default_factory=dict)
    total_proofs: int = 0

    def bump(self, position: int, rule: str) -> None:
        row = self.counts.setdefault(position, {})
        row[rule] = row.get(rule, 0) + 1

    def get(self, position: int, rule: str) -> int:
        return self.counts.get(position, {}).get(rule, 0)


def build_from_buffer(path: str) -> PositionRuleFreq:
    """Stream the buffer JSONL and tally per-position rule counts.

    Returns an empty counter if the file doesn't exist (first run with
    no buffer yet) — the sampler's Laplace smoothing handles that case
    correctly."""
    freq = PositionRuleFreq()
    if not os.path.exists(path):
        return freq
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            cert_toks = d.get("cert_toks")
            if not cert_toks:
                continue
            shape = cert_rule_shape(cert_toks)
            if not shape:
                continue
            for pos, rule in enumerate(shape):
                freq.bump(pos, rule)
            freq.total_proofs += 1
    return freq

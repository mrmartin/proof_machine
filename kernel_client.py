"""kernel_client.py — Python client for bin/verify_tokens.exe.

Speaks the line-oriented protocol used by the persistent OCaml kernel
worker.  Supports three request modes:

  1. Legacy full-cert verification:
        <cert_len> <cert_toks...> <goal_len> <goal_toks...>
     Response: a single integer (100 = ok, 0 = wrong concl, -1 = reject).

  2. H-prefixed full-cert verification:
        H <n_cn> <cn0..> <n_gn> <gn0..> <cert_len> <cert_toks...> <goal_len> <goal_toks...>
     Response: same as legacy.  Used when the goal references registered
     constants or named axioms whose original strings must be supplied.

  3. Prefix mode (NEW, added for M0):
        P <k> [H ...] <cert_len> <cert_toks...>
     Response:
        OK <num_thms> <step_id> <n_hyps>
              <hyp_0_len> <hyp_0_toks...> ... <concl_len> <concl_toks...>
              <step_id> <n_hyps> ... (repeated num_thms times)
     OR
        ERR <failing_step_index> [<message>]

The prefix-mode response is parsed into a [PrefixResult].

This is the M0 interface that M1 (tree search) and M3 (state exposure)
both build on.
"""
from __future__ import annotations

import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VERIFIER_BIN = os.path.join(
    _HERE, "_build", "default", "bin", "verify_tokens.exe"
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class ThmRecord:
    """One row of the kernel's theorem table after a verified prefix."""
    step_id: int
    hyps_toks: List[List[int]]       # token IDs per hypothesis (encoder pool)
    concl_toks: List[int]            # token IDs for the conclusion


@dataclass
class PrefixResult:
    ok: bool
    failing_step: int                # -1 on parse/decode failures; index on rule rejects
    message: str = ""                # OCaml-side reason string (best-effort)
    theorems: List[ThmRecord] = None # populated iff ok

    def __post_init__(self):
        if self.theorems is None:
            self.theorems = []


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------
def _parse_prefix_response(line: str) -> PrefixResult:
    parts = line.strip().split()
    if not parts:
        return PrefixResult(ok=False, failing_step=-1, message="empty_response")
    head = parts[0]
    if head == "ERR":
        # ERR <step_index> [message...]
        try:
            step_i = int(parts[1])
        except (IndexError, ValueError):
            step_i = -1
        msg = " ".join(parts[2:]) if len(parts) > 2 else ""
        return PrefixResult(ok=False, failing_step=step_i, message=msg)
    if head != "OK":
        return PrefixResult(ok=False, failing_step=-1,
                            message=f"bad_head: {head}")
    # OK <num_thms> <step_id> <n_hyps> <hyp_len> <hyp_toks...> ... <concl_len> <concl_toks...>
    i = 1
    try:
        num_thms = int(parts[i]); i += 1
        thms: List[ThmRecord] = []
        for _ in range(num_thms):
            step_id = int(parts[i]); i += 1
            n_hyps = int(parts[i]); i += 1
            hyps: List[List[int]] = []
            for _ in range(n_hyps):
                hlen = int(parts[i]); i += 1
                hyps.append([int(parts[i + j]) for j in range(hlen)])
                i += hlen
            clen = int(parts[i]); i += 1
            concl = [int(parts[i + j]) for j in range(clen)]
            i += clen
            thms.append(ThmRecord(step_id=step_id,
                                  hyps_toks=hyps, concl_toks=concl))
        return PrefixResult(ok=True, failing_step=-1, theorems=thms)
    except (IndexError, ValueError) as e:
        return PrefixResult(ok=False, failing_step=-1,
                            message=f"parse_failed: {e}")


# ---------------------------------------------------------------------------
# Single-process client
# ---------------------------------------------------------------------------
class KernelClient:
    """Owns one persistent OCaml verifier subprocess.

    Not thread-safe; wrap in [KernelClientPool] for concurrency."""
    def __init__(self, binary_path: str = DEFAULT_VERIFIER_BIN):
        if not os.path.exists(binary_path):
            raise FileNotFoundError(
                f"verifier binary missing: {binary_path}.  Run `make build`.")
        self._bin = binary_path
        self._proc: Optional[subprocess.Popen] = None
        self._spawn()

    def _spawn(self):
        if self._proc is not None:
            try: self._proc.kill()
            except Exception: pass
        self._proc = subprocess.Popen(
            [self._bin],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=1, universal_newlines=True,
        )

    def _send(self, req: str) -> str:
        try:
            self._proc.stdin.write(req)
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("verifier EOF")
            return line
        except Exception:
            self._spawn()
            raise

    @staticmethod
    def _format_names(names: Optional[Sequence[str]]) -> List[str]:
        ns = list(names or [])
        return [str(len(ns))] + ns

    def verify(self, cert_toks: Sequence[int], goal_toks: Sequence[int],
               cert_names: Optional[Sequence[str]] = None,
               goal_names: Optional[Sequence[str]] = None) -> int:
        parts: List[str] = []
        if cert_names is not None or goal_names is not None:
            parts += ["H"] + self._format_names(cert_names) \
                          + self._format_names(goal_names)
        parts += [str(len(cert_toks))] + [str(t) for t in cert_toks] \
               + [str(len(goal_toks))] + [str(t) for t in goal_toks]
        req = " ".join(parts) + "\n"
        try:
            line = self._send(req)
            return int(line.strip())
        except Exception:
            return -1

    def verify_prefix(self, cert_toks: Sequence[int], k: int,
                      cert_names: Optional[Sequence[str]] = None
                      ) -> PrefixResult:
        parts: List[str] = ["P", str(k)]
        if cert_names is not None:
            # Prefix mode reuses the H-prefix slot for cert names; the
            # goal-names slot is required by the parser but unused.
            parts += ["H"] + self._format_names(cert_names) \
                          + self._format_names(None)
        parts += [str(len(cert_toks))] + [str(t) for t in cert_toks]
        req = " ".join(parts) + "\n"
        try:
            line = self._send(req)
        except Exception as e:
            return PrefixResult(ok=False, failing_step=-1,
                                message=f"transport: {e}")
        return _parse_prefix_response(line)

    def close(self):
        if self._proc is not None:
            try: self._proc.kill()
            except Exception: pass
            self._proc = None


# ---------------------------------------------------------------------------
# Thread-pool wrapper for batched calls
# ---------------------------------------------------------------------------
class KernelClientPool:
    """Pool of persistent KernelClient threads.  Drop-in replacement for
    VerifierPool that additionally exposes verify_prefix_batch."""
    def __init__(self, n_threads: int,
                 binary_path: str = DEFAULT_VERIFIER_BIN):
        self._bin = binary_path
        self._tls = threading.local()
        self._exec = ThreadPoolExecutor(
            max_workers=n_threads, initializer=self._init_thread)
        self._n = n_threads

    def _init_thread(self):
        self._tls.client = KernelClient(self._bin)

    def _verify_one(self, cert_toks, goal_toks, cn, gn):
        return self._tls.client.verify(cert_toks, goal_toks, cn, gn)

    def _verify_prefix_one(self, cert_toks, k, cn):
        return self._tls.client.verify_prefix(cert_toks, k, cn)

    def verify_batch(self, jobs) -> List[int]:
        futs = []
        for job in jobs:
            if len(job) == 2:
                futs.append(self._exec.submit(
                    self._verify_one, job[0], job[1], None, None))
            else:
                futs.append(self._exec.submit(
                    self._verify_one, job[0], job[1], job[2], job[3]))
        return [f.result() for f in futs]

    def verify_prefix_batch(self, jobs) -> List[PrefixResult]:
        """jobs: list of (cert_toks, k) or (cert_toks, k, cert_names)."""
        futs = []
        for job in jobs:
            if len(job) == 2:
                futs.append(self._exec.submit(
                    self._verify_prefix_one, job[0], job[1], None))
            else:
                futs.append(self._exec.submit(
                    self._verify_prefix_one, job[0], job[1], job[2]))
        return [f.result() for f in futs]

    def close(self):
        self._exec.shutdown(wait=False, cancel_futures=True)

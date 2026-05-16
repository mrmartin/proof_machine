# How do I... verify a proof?

**Persona.** You have been handed two files:

- `something.kf` — the **theorem statement** in the kernel's logic, plus
  any axioms the proof depends on.
- `something.cert` — the **certificate**: a list of inference steps
  that supposedly proves the theorem.

You want the trusted kernel verifier $V$ to tell you whether the
certificate really proves the theorem. If $V$ says **accept**, the
theorem is true (modulo any declared axioms). If $V$ says **reject**,
the certificate is broken — possibly innocently (a typo), possibly
maliciously (someone is trying to claim a non-theorem). Either way,
you do not have to trust the certificate's author, only the kernel.

This is the simplest persona. You write no code.

---

## What you need

```
make build       # builds the binaries the first time
```

The verifier binary is at `_build/default/kernel/kernel_main.exe`.
It takes exactly two arguments:

```
vrfy  <theorem.kf>  <proof.cert>
```

Exit code 0 = accept; non-zero = reject (with a diagnostic on
stderr).

---

## Worked example: verify Euclid

The repo ships with a sample theorem (Euclid's infinitude of primes)
and a verified certificate.

### Step 1 — produce the artifacts

```
./_build/default/frontend/elab_main.exe \
    theories/number_theory/theory.thy euclid /tmp/euclid.kf

./_build/default/provers/pipeline_main.exe \
    --using scripted /tmp/euclid.kf /tmp/euclid.cert
```

(These two commands belong to other personas. If you were handed the
.kf and .cert files by a colleague, you skip them.)

### Step 2 — verify

```
./_build/default/kernel/kernel_main.exe /tmp/euclid.kf /tmp/euclid.cert
```

Expected output:

```
accept
```

Exit code is `0`. The 12-step certificate has been re-checked by the
kernel, and the theorem

$$\forall n : \mathbb{N}.\ \exists p : \mathbb{N}.\ p > n \wedge \text{prime}(p)$$

is provable from the declared axioms `has_prime_divisor` and
`euclid_step`.

---

## Worked example: see V reject something

The repo also ships with adversarial certificates that V **must**
reject. Each one is a small, hand-crafted lie about what a kernel
rule can do.

```
./_build/default/kernel/kernel_main.exe \
    tests/adversarial/phi.kf tests/adversarial/wrong_concl.cert
```

Expected output (on stderr) — and exit code `1`:

```
vrfy: REJECT — final derived (x = x) ≠ cert's declared concl (x = y)
```

Try the others (`bad_premise.cert`, `bad_witness.cert`,
`forge_axiom.cert`) the same way — each one will be rejected, each
for a different reason.

---

## What V is actually checking

For each step in the certificate, V replays the named primitive rule
on the named premises with the named witness and confirms the rule
returns the conclusion the step claims. If any rule call fails, the
certificate is rejected. After the last step, V also checks that the
derived theorem matches the goal in the `.kf` file.

Everything else — the certificate's author, the prover that produced
it, the layer that elaborated the .kf, the colour of the screen the
proof was rendered on — is **untrusted**. The whole point of the V/P
split is that *the only thing you need to trust is V*. See
[ARCHITECTURE.md](../ARCHITECTURE.md).

---

## When V rejects, what next?

- **`final derived X ≠ cert's declared concl Y`** — the certificate
  *claims* to prove Y but its steps actually conclude X. Either the
  author mis-wrote the `(concl ...)` line, or the proof is bogus.
- **`cert's concl X ≠ stated goal Y`** — the certificate proves X but
  you asked for Y. Either the wrong .kf was supplied, or the
  certificate is for a different theorem.
- **`step N (RULE_NAME): rule rejected: ...`** — step N is not a
  valid application of `RULE_NAME`. The diagnostic explains which side
  condition failed; consult [LOGIC.md](../LOGIC.md) for what each rule
  requires.
- **`step N cites unknown premise M`** — the step references a step
  that does not exist in the certificate.
- **`Axioms.lookup: no declared axiom named X`** — the certificate
  cited an axiom that the `.kf` file did not declare. Make sure the
  .kf and .cert come from the same theory package.

---

## Where to look next

- [`kernel/verify.ml`](../../kernel/verify.ml) — the entire trusted
  verifier (about 50 lines). You can read it in one sitting.
- [`docs/CERTIFICATE.md`](../CERTIFICATE.md) — the grammar of `.cert`
  files. Useful if you want to inspect what a certificate is doing.
- [`docs/LOGIC.md`](../LOGIC.md) — the primitive rules. Useful when
  V rejects a step and you want to understand why.

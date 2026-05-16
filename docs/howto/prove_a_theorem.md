# How do I... prove a theorem?

**Persona.** You have a `.kf` file stating a kernel formula $\varphi$
(maybe one you wrote yourself; see
[Formalize a theorem](formalize_a_theorem.md)). You want to produce
a `.cert` file that V will accept. Two routes:

1. **Run an existing prover.** Cheap, automatic, may fail.
2. **Hand-write the certificate.** Always possible, requires
   understanding kernel rules; you become the prover.

This guide does both. No OCaml in route 1; you only write S-expressions
in route 2.

---

## Route 1 — run the prover pipeline

The pipeline tries each named prover in order, takes the first
certificate the kernel accepts.

```
./_build/default/provers/pipeline_main.exe \
    --using <prover1>,<prover2>,...  <phi.kf>  <out.cert>
```

The MVP ships with three provers:

| Name         | Tries to prove                                 |
|--------------|------------------------------------------------|
| `lookup`     | already-cached certificates (sha-keyed by φ)   |
| `enumerator` | trivial identities like $\vdash t = t$ via REFL |
| `scripted`   | Euclid's infinitude of primes (hard-coded)     |

You can pass them in any order. The first one to emit a kernel-
accepted certificate wins.

### Example: prove the tiny REFL theorem

```
./_build/default/frontend/elab_main.exe \
    examples/tiny/refl.thy refl /tmp/refl.kf

./_build/default/provers/pipeline_main.exe \
    --using enumerator /tmp/refl.kf /tmp/refl.cert
```

Expected output: `prove: enumerator emitted certificate -> /tmp/refl.cert`.
Verify it:

```
./_build/default/kernel/kernel_main.exe /tmp/refl.kf /tmp/refl.cert
# → accept
```

### When none of the provers succeed

Output: `prove: no prover produced an accepted certificate`.

That means either (a) your goal really is not provable from the
declared axioms, or (b) none of the registered provers happens to
know how to prove it. The kernel cannot tell which; the kernel only
knows whether the certificate it was handed is valid. You can:

- Add another prover ([Write a theorem prover](write_a_theorem_prover.md)).
- Add a helper axiom in your theory package — but then you are
  taking it on faith; the renderer will list it as an axiom.
- Hand-write the certificate. Continue below.

---

## Route 2 — hand-write a certificate

A certificate is a sequence of inference-rule applications. You write
them as S-expressions in a `.cert` file:

```
(cert
  (step <id> (rule <RULE_NAME>) (witness <data>) (premises <id> <id> ...))
  ...
  (concl "<the final theorem in kernel-surface form>"))
```

Each step says "apply rule R to the theorems already at the cited
step ids, with this witness, and you'll get this conclusion". V
replays the script and accepts iff every rule call succeeds and the
last step's conclusion matches the goal.

The full list of rules and what each needs is in
[`docs/LOGIC.md`](../LOGIC.md). The certificate grammar in detail is
in [`docs/CERTIFICATE.md`](../CERTIFICATE.md).

### Worked example: prove `p ⇒ p`

Goal: for any proposition `p`, $\vdash p \Rightarrow p$.

The strategy is:

1. Assume `p` — get `{p} ⊢ p`.
2. Discharge the assumption — get `⊢ p ⇒ p`.

### Step 1 — write the statement

```
mkdir -p /tmp/proving
cat > /tmp/proving/id_imp.kf <<'EOF'
(goal "(! p : bool. (p:bool ==> p:bool))")
EOF
```

(The goal universally quantifies over the proposition `p` so it is
genuinely a closed kernel theorem.)

### Step 2 — write the certificate

Strategy in kernel rules:

| Step | Rule       | Yields                                            |
|------|------------|---------------------------------------------------|
| 1    | `ASSUME`   | `{p} ⊢ p`                                         |
| 2    | `DISCH`    | `⊢ p ⇒ p` (discharge the hypothesis `p`)          |
| 3    | `GEN`      | `⊢ ∀p. p ⇒ p`                                     |

For DISCH, the witness is the hypothesis to discharge (here: `p`).
For GEN, the witness is the variable to generalise over (here: `p`).

```
cat > /tmp/proving/id_imp.cert <<'EOF'
(cert
  (step 1 (rule ASSUME) (witness (term "p:bool")) (premises))
  (step 2 (rule DISCH)  (witness (term "p:bool")) (premises 1))
  (step 3 (rule GEN)    (witness (var "p" "bool")) (premises 2))
  (concl "(! p : bool. (p:bool ==> p:bool))"))
EOF
```

### Step 3 — run V

```
./_build/default/kernel/kernel_main.exe \
    /tmp/proving/id_imp.kf /tmp/proving/id_imp.cert
# → accept
```

If V rejects, the error message tells you which step failed.

---

## Larger example: read the scripted Euclid prover

For a much bigger hand-coded certificate, look at
[`provers/scripted.ml`](../../provers/scripted.ml). It emits a 12-step
certificate for Euclid's infinitude of primes using these rules:

```
AXIOM, SPEC, ASSUME, CONJUNCT1, MP, CONJ, EXISTS, CHOOSE, GEN
```

If you can read that file and understand each step, you can prove
anything the kernel admits — that is exactly what a tactic engine or
a neural prover ultimately reduces to.

---

## Witness syntax cheat-sheet

Most rules need a small piece of data alongside their premises:

| Witness form                                            | Used by                          |
|---------------------------------------------------------|----------------------------------|
| `(witness ())`                                          | rules with no extra data         |
| `(witness (term "<term>"))`                             | `REFL`, `BETA`, `ASSUME`, `SPEC`, `DISCH` |
| `(witness (type "<type>"))`                             | `INST_TYPE`                      |
| `(witness (var "<n>" "<type>"))`                        | `ABS`, `GEN`, `CHOOSE`           |
| `(witness (bound_and_witness (bound "<n>" "<type>") (witness "<term>")))` | `EXISTS`     |
| `(witness (axiom "<name>"))`                            | `AXIOM`                          |

`<term>`s are written in the same surface syntax as in `.thy` files
(see [Formalize a theorem](formalize_a_theorem.md)).

---

## Common rejection reasons (when you hand-write)

- **`SPEC: type mismatch`** — the term you supplied as the witness
  for SPEC has a different type than the universal's bound variable.
- **`MP: antecedent does not match`** — the consequent step says
  `A ⊢ p ⇒ q`, but the second premise's conclusion is not
  alpha-equivalent to `p`.
- **`CHOOSE: body theorem does not assume P[witness]`** — the body
  theorem must have `P[witness]` as a hypothesis (created via
  `ASSUME`) before you can `CHOOSE` it away.
- **`GEN: variable free in hypotheses`** — to universally generalise
  over `x`, `x` must not appear free in any hypothesis. Either
  discharge the offending hypothesis first (`DISCH`), or pick a
  fresh variable.

---

## Where to look next

- [Write a theorem prover](write_a_theorem_prover.md) — automate the
  certificate-emitting work so users do not have to hand-write.
- [`docs/LOGIC.md`](../LOGIC.md) — the canonical list of every rule
  and its side conditions.
- [`provers/scripted.ml`](../../provers/scripted.ml) — a real
  worked proof emitting a 12-step certificate.

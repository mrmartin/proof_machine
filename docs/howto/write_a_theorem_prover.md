# How do I... write a theorem prover?

**Persona.** You have an *idea* for how to find proofs — a tactic
engine, a transformer that emits proof scripts, a SAT solver with
proof traces, a Monte Carlo tree search, a lookup table over
Mathlib hashes — and you want to plug it into proof_machine.

The plugin contract is one OCaml `module type`: you implement
`prove : φ → Stream<Cert>`. The kernel re-verifies every certificate
your prover emits, so you cannot break soundness no matter how
buggy or hallucinatory your code is. That is the whole architectural
point of the V/P split.

This persona writes OCaml. Basic familiarity is enough.

---

## The plugin contract

In [`provers/prover_api.ml`](../../provers/prover_api.ml):

```ocaml
module type PROVER = sig
  val name : string
  val prove :
    phi:Kernel.Term.term ->
    budget:budget ->
    hints:hint list ->
    Kernel.Cert.t Seq.t
end
```

- `name` — a short identifier used on the command line
  (`--using <name>`).
- `prove ~phi ~budget ~hints` — produce a (possibly empty, possibly
  infinite) lazy sequence of candidate certificates for the goal
  `phi`. You can return them in any order.

The pipeline driver iterates through your sequence, calls V on each
certificate, and takes the first one V accepts. Bad certificates are
silently discarded.

---

## Worked example: write a `tautology` prover that proves `p ⇒ p`

Goal: any φ of the form $\forall p : \text{bool}.\ p \Rightarrow p$
(up to alpha-equivalence) should be provable by your new prover.

### Step 1 — create the file

```
cat > provers/tautology.ml <<'EOF'
(* provers/tautology.ml — a tiny prover that handles ∀p:bool. p ⇒ p. *)

open Kernel

let name = "tautology"

(* Recognise goals alpha-equivalent to ∀p:bool. p ⇒ p. *)
let is_self_imp phi =
  let p = Term.Var ("p", Type.bool_ty) in
  let target =
    Rules.mk_forall ("p", Type.bool_ty) (Rules.mk_imp p p)
  in
  Term.alpha_eq phi target

(* Build the three-step certificate by hand. *)
let cert_for_self_imp () =
  let p = Term.Var ("p", Type.bool_ty) in
  let mk id rule witness premises =
    { Cert.id; rule; witness; premises; declared_concl = None }
  in
  let steps = [
    mk 1 "ASSUME" (Cert.W_term p)             [];
    mk 2 "DISCH"  (Cert.W_term p)             [1];
    mk 3 "GEN"    (Cert.W_var ("p", Type.bool_ty)) [2];
  ] in
  { Cert.steps; concl = Rules.mk_forall ("p", Type.bool_ty) (Rules.mk_imp p p) }

let prove ~phi ~budget:_ ~hints:_ =
  if is_self_imp phi then Seq.return (cert_for_self_imp ())
  else Seq.empty
EOF
```

### Step 2 — register the module in dune and the pipeline

Edit [`provers/dune`](../../provers/dune) to add `tautology` to the
library's module list:

```
(library
 (name provers)
 (wrapped true)
 (modules prover_api lookup enumerator scripted tautology)
 (libraries kernel))
```

And edit [`provers/pipeline_main.ml`](../../provers/pipeline_main.ml)
to add an entry in the `registered` table:

```ocaml
let registered = [
  ("lookup",     fun ~phi ~budget ~hints ->
     Provers.Lookup.prove ~phi ~budget ~hints);
  ("scripted",   fun ~phi ~budget ~hints ->
     Provers.Scripted.prove ~phi ~budget ~hints);
  ("tautology",  fun ~phi ~budget ~hints ->
     Provers.Tautology.prove ~phi ~budget ~hints);   (* NEW *)
  ("enumerator", fun ~phi ~budget ~hints ->
     Provers.Enumerator.prove ~phi ~budget ~hints);
]
```

### Step 3 — build

```
dune build
```

### Step 4 — try it

Write the theorem and goal file:

```
mkdir -p /tmp/taut
cat > /tmp/taut/taut.thy <<'EOF'
theorem self_imp : (! p : bool. (p:bool ==> p:bool))
EOF

./_build/default/frontend/elab_main.exe \
    /tmp/taut/taut.thy self_imp /tmp/taut/taut.kf

./_build/default/provers/pipeline_main.exe \
    --using tautology /tmp/taut/taut.kf /tmp/taut/taut.cert
# → prove: tautology emitted certificate -> /tmp/taut/taut.cert

./_build/default/kernel/kernel_main.exe \
    /tmp/taut/taut.kf /tmp/taut/taut.cert
# → accept
```

Done. You wrote a prover. The kernel did not change.

---

## What just happened

You added ~25 lines of OCaml and one entry in a dispatch table. The
kernel cannot see your prover at all; it only sees the certificate.
If your prover had emitted

- a certificate for a different theorem — V would reject it and the
  pipeline would move on to the next prover.
- a certificate that fabricates a premise — V would reject it.
- a certificate that uses a rule incorrectly — V would reject it.

None of these failures can produce a false theorem. Untrusted code
plus a trusted verifier = soundness preserved.

---

## Going further

Real-world provers are mostly variations on these shapes:

### Pattern: lookup / cache

See [`provers/lookup.ml`](../../provers/lookup.ml). Hash the
canonical form of φ, check a disk cache, return the cached cert if
hit. This is the single most useful prover for working
mathematicians: most theorems are *re*-proved, not freshly proved.

### Pattern: shape-driven scripting

See [`provers/scripted.ml`](../../provers/scripted.ml). Match φ's
syntactic shape; if it fits a pattern you know how to discharge,
emit a hand-written certificate skeleton. This is also what a tactic
language like Ltac compiles down to.

### Pattern: search

See [`provers/enumerator.ml`](../../provers/enumerator.ml) for the
stub. A real search prover does BFS or A* through rule applications.
Be sure to lazily yield candidates via `Seq` so the pipeline can
stop pulling once V accepts.

### Pattern: external oracle

Shell out to a SAT/SMT solver, an LLM, or a remote service. Parse
the response into kernel rule applications. The kernel only sees
what you emit; the external oracle is fully untrusted.

```ocaml
let prove ~phi ~budget ~hints:_ =
  let response = Http.post "https://my-llm/prove" ~json:(json_of phi) in
  let candidate = decode response in
  Seq.return candidate    (* lazy: only built if pulled *)
```

### Pattern: parallel race

You can compose provers by wrapping them in another `PROVER`-shape
module that interleaves multiple internal sources. The pipeline
already tries provers in series; for racing, write one prover that
itself manages the parallelism.

---

## Constraints to keep in mind

- **Soundness is V's job, not yours.** Do not try to verify your own
  output internally; that just bloats your code. Emit and move on.
- **Stay lazy.** Returning an infinite `Seq` is fine. Returning an
  eagerly-computed list of a million certificates is not.
- **Time and memory budgets are advisory.** The pipeline does not
  currently enforce `budget`; you must respect it yourself.
- **The kernel's `Term.term` is opaque.** Use the constructors in
  [`kernel/term.ml`](../../kernel/term.ml) (e.g. `Term.mk_comb`,
  `Term.mk_abs`, `Term.mk_eq`) and the connective helpers in
  [`kernel/rules.ml`](../../kernel/rules.ml) (`Rules.mk_forall`,
  `Rules.mk_exists`, `Rules.mk_conj`, `Rules.mk_imp`). Do not
  construct `Thm.t` values — only the kernel can.

---

## Where to look next

- [`docs/PROVER_API.md`](../PROVER_API.md) — the formal plugin
  contract.
- [`provers/scripted.ml`](../../provers/scripted.ml) — a worked
  example: a 12-step Euclid certificate, programmatically generated.
- [Write a kernel](write_a_kernel.md) — when even more flexibility
  is wanted *inside* the trust boundary (e.g., a new logic).

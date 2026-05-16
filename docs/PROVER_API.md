# Prover plugin API

A prover is any program matching the signature

```ocaml
module type PROVER = sig
  val name : string
  val prove :
    phi:Term.t ->
    ctx:Library.t ->
    budget:Resources.t ->
    hints:Hint.t list ->
    Cert.t Seq.t
end
```

That's it. The prover may:

- be deterministic or randomised
- terminate or loop forever (the kernel stops pulling on the stream
  when `budget` is exhausted)
- be sound or unsound; the kernel filters
- run in process, fork to a GPU, hit a remote API, or shell out to a
  SAT solver

A provider implements `PROVER`, links against `kernel`, and registers
itself via `Pipeline.register`. The pipeline driver
(`provers/pipeline_main.ml`) iterates through the user-selected
providers, takes the first certificate the kernel accepts, and returns
it. Failed certificates are discarded silently.

## Reference implementations

- `provers/lookup/` — sha256 of canonical-form φ → cached cert. Useful
  for already-proved lemmas.
- `provers/enumerator/` — naive bounded BFS through Hilbert proofs.
  Pedagogical baseline; proves tiny logical identities.
- `provers/scripted/` — emits a hand-coded certificate for known φ
  (currently: Euclid's infinitude of primes). Stands in for what a
  tactic engine would do.

## Adding a new prover

1. Create `provers/<name>/dune` and `provers/<name>/<name>.ml`
   implementing `Prover_api.PROVER`.
2. Add `(provers/<name>)` to `provers/pipeline_main.ml`'s registration
   list.
3. The kernel needs no changes. Build and use.

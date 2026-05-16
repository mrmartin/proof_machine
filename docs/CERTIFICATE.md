# Certificate format

A certificate is a sequence of inference steps, each of which records
which primitive rule was applied, what premises (by step id) it took,
what witnesses (terms, types) it needed, and what theorem it
produced. The last step's conclusion is compared against the
externally-supplied φ; success means **V accepts**.

## Grammar

```
cert    := (cert <step>* <concl>)
step    := (step <id> (rule <Name>) (witness <wit>) (premises <id>*))
wit     := ()                          -- rule needs no witness
         | (term  "<HOL term>")        -- e.g. for REFL, ASSUME, INST
         | (type  "<HOL type>")        -- e.g. for INST_TYPE
         | (var   "<name>" "<type>")   -- e.g. for ABS
         | (inst  ((name "<v>" "<t>"))*) -- substitution lists for INST
         | (insttype ((tyvar "<a>" "<T>"))*)
concl   := (concl "<HOL term>")
```

Where:

- `<id>` is a positive integer, unique within the certificate.
- `<HOL term>` is the textual term syntax accepted by `Term.parse`.
- `<HOL type>` is the textual type syntax accepted by `Type.parse`.
- `<Name>` is one of `REFL TRANS MK_COMB ABS BETA ASSUME EQ_MP
  DEDUCT_ANTISYM_RULE INST INST_TYPE AXIOM`.

The pseudo-rule `AXIOM` admits a single witness of form
`(axiom "<name>")` and zero premises; the verifier consults a
manifest of declared axioms to mint the corresponding theorem. The
declared axioms are part of the trusted base for the package that
declares them; the renderer marks any step using `AXIOM` as such.

## Example (Euclid, abbreviated)

```
(cert
  (step 1  (rule AXIOM)    (witness (axiom "has_prime_divisor")) (premises))
  (step 2  (rule AXIOM)    (witness (axiom "fact_pos"))          (premises))
  …
  (step N  (rule MK_COMB)  (witness ())                          (premises a b))
  (concl "∀N:nat. ∃p:nat. (gt p N) ∧ prime p"))
```

## Verification

Pseudo-code; the actual implementation is in `kernel/verify.ml`:

```
let verify cert phi =
  let table = Hashtbl.create 64 in
  List.iter (fun step ->
    let prems = List.map (Hashtbl.find table) step.premises in
    let thm   = Rules.apply step.rule step.witness prems in
    Hashtbl.add table step.id thm
  ) cert.steps;
  Term.alpha_eq (Thm.concl (Hashtbl.find table cert.last)) phi
```

Time complexity: O(|cert| · L) where L bounds term size. The verifier
holds no global state across runs.

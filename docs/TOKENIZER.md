# HOL certificate tokenizer

Lexeme-level vocabulary for HOL terms, types, and certificates.  Sized to
fit in **uint8** (vocab_size = 232) so the token stream stores directly
as `np.uint8` and is compatible with vanilla GPT loops.

Two implementations that agree byte-for-byte:

- **OCaml library** under `tokenizer/` — used by `bin/verify_tokens.exe`
  and as the kernel-side reference.
- **Python port** at `tokenizer/python/hol_tokenizer.py` — used by the
  trainer in `microgpt_with_RL_hol.py`.

## Vocab layout (see `lexicon.ml` / `hol_tokenizer.py` for exact IDs)

```
  0..3     special   : BOS, EOS, PAD, UNK
  4..15    structural: ( ) : . , " ->  (+ reserved)
 16..23    term ops  : = ==> /\ \/ ~ ! ? \
 24..47    keywords  : type, const, axiom, theorem, goal, cert, step,
                       rule, witness, premises, concl, term, var, inst,
                       insttype, bound_and_witness, subst, bound
 48..71    rule names: REFL, TRANS, MK_COMB, ABS, BETA, ASSUME, EQ_MP,
                       DEDUCT_ANTISYM_RULE, INST, INST_TYPE, AXIOM, GEN,
                       SPEC, EXISTS, CHOOSE, CONJ, CONJUNCT1, CONJUNCT2,
                       MP, DISCH, ETA_AX, SELECT_AX, EM_AX
 72..79    builtin ty: bool, ind, fun, nat (+ reserved)
 80..87    type-var slots: 'a, 'b, ... (canonical pool)
 88..103   type-ctor slots (theory-declared)
104..167   name slots (constants + axiom names, theory-declared)
168..199   variable slots (canonicalised per cert)
200..231   integer literal slots (step ids, premise ids)
```

Open-vocab classes (variables, names, type-ctors, integer literals) are
mapped to fixed-size **pools**.  The encoder allocates the next free slot
on first sight; subsequent occurrences of the same name reuse the slot.
This collapses alpha-equivalent occurrences and bounds the vocabulary at
encode time.

## Grammar PDA

`tokenizer/grammar.ml` (and its Python port) implements a deterministic
pushdown automaton over the token stream.  Its language is exactly the
image of the encoder — for any `Cert.t` accepted by the OCaml kernel,
the encoded token sequence is accepted by the PDA, and `is_accepting`
fires at the closing `)`.

API:

```ocaml
val initial         : state
val step            : state -> int -> state option   (* None = reject *)
val is_accepting    : state -> bool
val valid_next_mask : state -> bool array            (* len = vocab_size *)
```

The mask is **sound** (never excludes a legal continuation) and tight
enough on structural positions that a masked-sample policy is forced
into well-formed cert prefixes.

## What's tested

`dune test` runs three properties over the bundled Euclid certificate,
200 synthesised certs (see `tokenizer/synth.ml`), and the tiny REFL toy:

1. `decode (encode c) ≈α c` — round-trip up to alpha-equivalence.
2. Every encoded stream is accepted by the grammar PDA and reaches
   `is_accepting`.
3. Mask soundness — at every prefix of every encoded example, the
   actual next token is in `valid_next_mask(state)`.

Total: 609 PASS / 0 FAIL on the current main.

## Why two implementations

The OCaml side decodes back into the trusted kernel's `Term.term` and
`Cert.t`, so `Verify.verify` can run on whatever the trainer produced.

The Python side does the same encode/decode/mask logic in
≈200 LOC — the trainer needs the mask at every generation step, so
calling out to OCaml per token would dominate the run time.  The two
implementations are kept in lockstep by hand (small enough that this is
fine in practice; the test in `tokenizer/python/test_tokenizer.py` does
a sanity round-trip).

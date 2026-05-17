"""Sanity test for the Python port: round-trip + grammar acceptance on a
handful of hand-built certs."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import hol_tokenizer as T


def build_refl_cert(name="x"):
    nat = T.nat_ty()
    x = T.mk_var(name, nat)
    return T.Cert(
        steps=[T.Step(1, "REFL", ("term", x), [])],
        concl=T.mk_eq(x, x),
    )


def build_imp_self(name="p"):
    p = T.mk_var(name, T.bool_ty())
    return T.Cert(
        steps=[
            T.Step(1, "ASSUME", ("term", p), []),
            T.Step(2, "DISCH",  ("term", p), [1]),
        ],
        concl=T.mk_imp(p, p),
    )


def build_forall_refl(name="x"):
    nat = T.nat_ty()
    x = T.mk_var(name, nat)
    return T.Cert(
        steps=[
            T.Step(1, "REFL", ("term", x), []),
            T.Step(2, "GEN",  ("var", name, nat), [1]),
        ],
        concl=T.mk_forall(name, nat, T.mk_eq(x, x)),
    )


def check_roundtrip(label, cert):
    toks, hdr = T.encode_cert(cert)
    dec = T.decode_cert(hdr, toks)
    ok = T.alpha_eq(cert.concl, dec.concl) and len(cert.steps) == len(dec.steps)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}  ({len(toks)} toks)")
    return ok


def check_grammar(label, cert):
    toks, _ = T.encode_cert(cert)
    s = T.initial_state()
    for i, t in enumerate(toks):
        ns = T.step(s, t)
        if ns is None:
            print(f"  FAIL  grammar {label}: rejected @ {i} ({T.tok_str(t)})")
            return False
        s = ns
    if not T.is_accepting(s):
        print(f"  FAIL  grammar {label}: not accepting (stack={s})")
        return False
    print(f"  PASS  grammar {label}")
    return True


def check_mask(label, cert):
    toks, _ = T.encode_cert(cert)
    s = T.initial_state()
    for i, t in enumerate(toks):
        m = T.valid_next_mask(s)
        if not m[t]:
            print(f"  FAIL  mask {label} @ {i}: tok {T.tok_str(t)} not in mask")
            return False
        s = T.step(s, t)
    print(f"  PASS  mask {label}")
    return True


def main():
    certs = [
        ("REFL x",     build_refl_cert()),
        ("p ⇒ p",      build_imp_self()),
        ("∀x. x = x",  build_forall_refl()),
    ]
    print(f"vocab_size = {T.VOCAB_SIZE}")
    all_ok = True
    print("Round-trip:")
    for n, c in certs: all_ok &= check_roundtrip(n, c)
    print("Grammar:")
    for n, c in certs: all_ok &= check_grammar(n, c)
    print("Mask:")
    for n, c in certs: all_ok &= check_mask(n, c)
    print()
    print("OK" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

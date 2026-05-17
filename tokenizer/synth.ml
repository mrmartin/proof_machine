(* tokenizer/synth.ml — generate synthetic Cert.t values that exercise
   the rule set the encoder must cover.

   The certificates here are constructed *syntactically* — we build
   Cert.step records directly without running the kernel verifier.
   Their purpose is to give the tokenizer tests a varied corpus, not to
   prove anything.  The terms they contain are well-formed HOL terms
   built via the kernel's smart constructors, so the encoder's pattern
   matches all fire correctly. *)

open Kernel

let bool_ty = Type.bool_ty
let nat_ty () = Type.Tyapp ("nat", [])
let () = if not (Type.well_formed (nat_ty ())) then
  Type.register_tyconstr "nat" 0

let var n ty = Term.Var (n, ty)

let mk id rule witness premises : Cert.step =
  { Cert.id; rule; witness; premises; declared_concl = None }

(* --- Templates ------------------------------------------------------- *)

(* template 1: ⊢ x = x where x is a fresh variable name *)
let refl_cert name =
  let t = var name (nat_ty ()) in
  let concl = Term.mk_eq t t in
  let steps = [
    mk 1 "REFL" (Cert.W_term t) []
  ] in
  { Cert.steps; concl }

(* template 2: {p} ⊢ p — single ASSUME *)
let assume_cert name =
  let p = var name bool_ty in
  let steps = [
    mk 1 "ASSUME" (Cert.W_term p) []
  ] in
  { Cert.steps; concl = p }

(* template 3: ⊢ p ⇒ p — ASSUME then DISCH *)
let imp_self_cert name =
  let p = var name bool_ty in
  let steps = [
    mk 1 "ASSUME" (Cert.W_term p) [];
    mk 2 "DISCH"  (Cert.W_term p) [1];
  ] in
  let concl = Rules.mk_imp p p in
  { Cert.steps; concl }

(* template 4: ⊢ ∀x. x = x — REFL + GEN *)
let forall_refl_cert name =
  let x = var name (nat_ty ()) in
  let steps = [
    mk 1 "REFL" (Cert.W_term x) [];
    mk 2 "GEN"  (Cert.W_var (name, nat_ty ())) [1];
  ] in
  let concl = Rules.mk_forall (name, nat_ty ()) (Term.mk_eq x x) in
  { Cert.steps; concl }

(* template 5: ⊢ ∀p. p ⇒ p — ASSUME, DISCH, GEN *)
let forall_imp_self_cert name =
  let p = var name bool_ty in
  let steps = [
    mk 1 "ASSUME" (Cert.W_term p) [];
    mk 2 "DISCH"  (Cert.W_term p) [1];
    mk 3 "GEN"    (Cert.W_var (name, bool_ty)) [2];
  ] in
  let concl = Rules.mk_forall (name, bool_ty) (Rules.mk_imp p p) in
  { Cert.steps; concl }

(* template 6: BETA  ⊢ (\x:nat. x) y = y, for a fresh outer free y *)
let beta_cert () =
  let body = Term.Var ("x", nat_ty ()) in
  let lam  = Term.mk_abs (Term.Var ("x", nat_ty ())) body in
  let y    = Term.Var ("y", nat_ty ()) in
  let app  = Term.mk_comb lam y in
  let steps = [
    mk 1 "BETA" (Cert.W_term app) []
  ] in
  let concl = Term.mk_eq app y in
  { Cert.steps; concl }

(* template 7: ⊢ p ∧ q ⇒ q ∧ p — uses CONJUNCT1/2, CONJ, DISCH *)
let conj_swap_cert pname qname =
  let p = var pname bool_ty in
  let q = var qname bool_ty in
  let pq = Rules.mk_conj p q in
  let qp = Rules.mk_conj q p in
  let steps = [
    mk 1 "ASSUME"    (Cert.W_term pq) [];
    mk 2 "CONJUNCT1" Cert.W_none [1];
    mk 3 "CONJUNCT2" Cert.W_none [1];
    mk 4 "CONJ"      Cert.W_none [3; 2];
    mk 5 "DISCH"     (Cert.W_term pq) [4];
  ] in
  let concl = Rules.mk_imp pq qp in
  { Cert.steps; concl }

(* template 8: ⊢ ∃x. x = x — REFL, EXISTS *)
let exists_refl_cert name =
  let nat = nat_ty () in
  let x = var name nat in
  let steps = [
    mk 1 "REFL"   (Cert.W_term x) [];
    mk 2 "EXISTS" (Cert.W_bound_and_witness ((name, nat), x)) [1];
  ] in
  let concl = Rules.mk_exists (name, nat) (Term.mk_eq x x) in
  { Cert.steps; concl }

(* template 9: SPEC of ∀x. x = x at a concrete y *)
let spec_cert name witness_name =
  let nat = nat_ty () in
  let x = var name nat in
  let y = var witness_name nat in
  let universal = Rules.mk_forall (name, nat) (Term.mk_eq x x) in
  let steps = [
    mk 1 "REFL" (Cert.W_term x) [];
    mk 2 "GEN"  (Cert.W_var (name, nat)) [1];
    mk 3 "SPEC" (Cert.W_term y) [2];
  ] in
  let _ = universal in
  let concl = Term.mk_eq y y in
  { Cert.steps; concl }

(* --- Generator ------------------------------------------------------- *)

(* A small pool of variable names, used round-robin to ensure pool
   slots stay within their budgets. *)
let names = [|
  "a"; "b"; "c"; "p"; "q"; "r"; "x"; "y"; "z"; "n"; "m"; "k"; "u"; "v"; "w"; "s"; "t"
|]

let pick rs = names.(Random.State.int rs (Array.length names))

let one rs =
  match Random.State.int rs 9 with
  | 0 -> refl_cert (pick rs)
  | 1 -> assume_cert (pick rs)
  | 2 -> imp_self_cert (pick rs)
  | 3 -> forall_refl_cert (pick rs)
  | 4 -> forall_imp_self_cert (pick rs)
  | 5 -> beta_cert ()
  | 6 ->
    let a = pick rs in
    let b =
      let b' = pick rs in
      if b' = a then names.((Array.length names - 1)) else b'
    in
    conj_swap_cert a b
  | 7 -> exists_refl_cert (pick rs)
  | _ -> spec_cert (pick rs) (pick rs)

let gen ?(seed = 0) ~n () =
  let rs = Random.State.make [| seed |] in
  List.init n (fun _ -> one rs)

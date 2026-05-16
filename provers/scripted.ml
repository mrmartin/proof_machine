(* provers/scripted.ml — a "scripted tactic" prover for Euclid.

   Recognises goals of the form
     ! n:nat. ? q:nat. (gt q n) /\ (prime q)
   (any alpha-equivalent variant) and emits a 12-step certificate
   that derives the goal from the declared number-theory axioms
   [has_prime_divisor] and [euclid_step], using the kernel's
   connective primitives. *)

open Kernel

let name = "scripted"

(* --- Shape check: is φ alpha-equivalent to Euclid's statement? ------- *)

let nat = Type.Tyapp ("nat", [])

let euclid_phi () =
  let n = Term.Var ("n", nat) in
  let q = Term.Var ("q", nat) in
  let gt = Term.Const ("gt", Type.fun_ty nat (Type.fun_ty nat Type.bool_ty)) in
  let prime = Term.Const ("prime", Type.fun_ty nat Type.bool_ty) in
  let body =
    Rules.mk_conj
      (Term.mk_comb (Term.mk_comb gt q) n)
      (Term.mk_comb prime q)
  in
  Rules.mk_forall ("n", nat)
    (Rules.mk_exists ("q", nat) body)

let is_euclid phi =
  try Term.alpha_eq phi (euclid_phi ()) with _ -> false

(* --- Emit Euclid's certificate -------------------------------------- *)

let euclid_cert () =
  let n_var = Term.Var ("n", nat) in
  let p_var = Term.Var ("p", nat) in
  let factorial =
    Term.Const ("factorial", Type.fun_ty nat nat) in
  let plus_c =
    Term.Const ("plus", Type.fun_ty nat (Type.fun_ty nat nat)) in
  let one = Term.Const ("one", nat) in
  let prime =
    Term.Const ("prime", Type.fun_ty nat Type.bool_ty) in
  let divides =
    Term.Const ("divides", Type.fun_ty nat (Type.fun_ty nat Type.bool_ty)) in
  let gt =
    Term.Const ("gt", Type.fun_ty nat (Type.fun_ty nat Type.bool_ty)) in
  let m_term =
    Term.mk_comb (Term.mk_comb plus_c (Term.mk_comb factorial n_var)) one in
  let h_term =
    Rules.mk_conj
      (Term.mk_comb prime p_var)
      (Term.mk_comb (Term.mk_comb divides p_var) m_term)
  in
  let mk id rule witness premises =
    { Cert.id; rule; witness; premises; declared_concl = None }
  in
  let steps : Cert.step list = [
    mk 1  "AXIOM"     (Cert.W_axiom "has_prime_divisor")          [];
    mk 2  "SPEC"      (Cert.W_term m_term)                        [1];
    mk 3  "ASSUME"    (Cert.W_term h_term)                        [];
    mk 4  "CONJUNCT1" Cert.W_none                                 [3];
    mk 5  "AXIOM"     (Cert.W_axiom "euclid_step")                [];
    mk 6  "SPEC"      (Cert.W_term p_var)                         [5];
    mk 7  "SPEC"      (Cert.W_term n_var)                         [6];
    mk 8  "MP"        Cert.W_none                                 [7; 3];
    mk 9  "CONJ"      Cert.W_none                                 [8; 4];
    mk 10 "EXISTS"
      (Cert.W_bound_and_witness (("q", nat), p_var))              [9];
    mk 11 "CHOOSE"    (Cert.W_var ("p", nat))                     [2; 10];
    mk 12 "GEN"       (Cert.W_var ("n", nat))                     [11];
  ] in
  let concl =
    Rules.mk_forall ("n", nat)
      (Rules.mk_exists ("q", nat)
         (Rules.mk_conj
            (Term.mk_comb (Term.mk_comb gt (Term.Var ("q", nat))) n_var)
            (Term.mk_comb prime (Term.Var ("q", nat)))))
  in
  { Cert.steps; concl }

let prove ~phi ~budget:_ ~hints:_ =
  if is_euclid phi then Seq.return (euclid_cert ())
  else Seq.empty

(* tests/test_kernel.ml — unit tests for the kernel primitives.

   Each test constructs a small theorem using the rules and checks the
   resulting conclusion is what we expect (up to alpha-equivalence).
   Failures are loud: any non-zero exit fails [dune runtest]. *)

open Kernel

let failed = ref 0
let passed = ref 0

let check name cond =
  if cond then (incr passed; Printf.printf "  PASS  %s\n" name)
  else begin
    incr failed;
    Printf.printf "  FAIL  %s\n" name
  end

let check_eq name t1 t2 =
  check name (Term.alpha_eq t1 t2)

let check_raises name f =
  match (try Some (f ()) with _ -> None) with
  | None -> incr passed; Printf.printf "  PASS  %s (raised)\n" name
  | Some _ -> incr failed; Printf.printf "  FAIL  %s (no raise)\n" name

let nat = Type.Tyapp ("nat", [])
let () = if not (Type.well_formed nat) then
  Type.register_tyconstr "nat" 0

let bool_ = Type.bool_ty

(* --- 10 primitive rules --- *)

let () =
  Printf.printf "Kernel primitive rules:\n";

  (* REFL *)
  let t = Term.Var ("x", nat) in
  let th = Rules.refl t in
  check_eq "REFL produces t = t"
    (Thm.concl th)
    (Term.mk_eq t t);

  (* TRANS *)
  let a = Term.Var ("a", nat) in
  let b = Term.Var ("b", nat) in
  let c = Term.Var ("c", nat) in
  let h1 = Rules.assume (Term.mk_eq a b) in
  let h2 = Rules.assume (Term.mk_eq b c) in
  let th = Rules.trans h1 h2 in
  check_eq "TRANS chains equalities"
    (Thm.concl th)
    (Term.mk_eq a c);
  check_raises "TRANS rejects mismatched middle"
    (fun () ->
       let bad = Rules.assume (Term.mk_eq c a) in
       Rules.trans h1 bad);

  (* MK_COMB *)
  let f = Term.Var ("f", Type.fun_ty nat nat) in
  let g = Term.Var ("g", Type.fun_ty nat nat) in
  let h1 = Rules.assume (Term.mk_eq f g) in
  let h2 = Rules.assume (Term.mk_eq a b) in
  let th = Rules.mk_comb h1 h2 in
  check_eq "MK_COMB applies congruence"
    (Thm.concl th)
    (Term.mk_eq (Term.mk_comb f a) (Term.mk_comb g b));

  (* BETA *)
  let v = Term.Var ("z", nat) in
  let body = Term.mk_comb (Term.Var ("g", Type.fun_ty nat nat)) v in
  let t = Term.mk_comb (Term.mk_abs v body) v in
  let th = Rules.beta t in
  check_eq "BETA reduces (λz.g z) z to g z"
    (Thm.concl th)
    (Term.mk_eq t body);

  (* ASSUME *)
  let p = Term.Var ("p", bool_) in
  let th = Rules.assume p in
  check "ASSUME tracks the hypothesis"
    (Thm.hyps th = [p] && Term.alpha_eq (Thm.concl th) p);
  check_raises "ASSUME rejects non-bool"
    (fun () -> Rules.assume (Term.Var ("k", nat)));

  (* EQ_MP *)
  let q = Term.Var ("q", bool_) in
  let h1 = Rules.assume (Term.mk_eq p q) in
  let h2 = Rules.assume p in
  let th = Rules.eq_mp h1 h2 in
  check_eq "EQ_MP rewrites" (Thm.concl th) q;

  (* INST *)
  let th0 = Rules.refl (Term.Var ("x", nat)) in
  let th = Rules.inst [(Term.Var ("x", nat), a)] th0 in
  check_eq "INST substitutes free variables"
    (Thm.concl th)
    (Term.mk_eq a a);

  (* GEN / SPEC roundtrip *)
  let p_var = Term.Var ("p", nat) in
  let th0 = Rules.refl p_var in
  let th_g = Rules.gen p_var th0 in
  let th_s = Rules.spec a th_g in
  check_eq "GEN/SPEC roundtrip"
    (Thm.concl th_s)
    (Term.mk_eq a a);

  (* CONJ / CONJUNCT *)
  let th1 = Rules.assume p in
  let th2 = Rules.assume q in
  let th_and = Rules.conj th1 th2 in
  check_eq "CONJ produces p /\\ q"
    (Thm.concl th_and)
    (Rules.mk_conj p q);
  check_eq "CONJUNCT1 extracts left"
    (Thm.concl (Rules.conjunct1 th_and)) p;
  check_eq "CONJUNCT2 extracts right"
    (Thm.concl (Rules.conjunct2 th_and)) q;

  (* MP / DISCH *)
  let pq = Rules.mk_imp p q in
  let th_pq = Rules.assume pq in
  let th_p = Rules.assume p in
  let th_q = Rules.mp th_pq th_p in
  check_eq "MP yields the consequent" (Thm.concl th_q) q;
  let th_disch = Rules.disch p th_q in
  check_eq "DISCH discharges and produces p ⇒ q"
    (Thm.concl th_disch)
    pq;

  Printf.printf "\nResults: %d passed, %d failed\n" !passed !failed;
  if !failed > 0 then exit 1

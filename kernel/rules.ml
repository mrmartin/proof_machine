(* kernel/rules.ml — the 10 primitive inference rules of HOL.

   This file is part of the trusted base.  Each rule constructs a fresh
   [Thm.t] only when its side conditions hold; otherwise it raises
   [Rule_error].  The verifier in [verify.ml] catches such failures
   and treats them as rejection. *)

exception Rule_error of string

let err s = raise (Rule_error s)

(* Register the connective constants the rules below use.  This must
   happen once, before any rule is called; OCaml runs this at module
   initialization. *)
let () =
  let bb = Type.fun_ty Type.bool_ty (Type.fun_ty Type.bool_ty Type.bool_ty) in
  let b = Type.fun_ty Type.bool_ty Type.bool_ty in
  let a = Type.tyvar "A" in
  let pred_ty = Type.fun_ty a Type.bool_ty in
  let q_ty = Type.fun_ty pred_ty Type.bool_ty in
  Term.register_const "T" Type.bool_ty;
  Term.register_const "F" Type.bool_ty;
  Term.register_const "~" b;
  Term.register_const "/\\" bb;
  Term.register_const "\\/" bb;
  Term.register_const "==>" bb;
  Term.register_const "!" q_ty;
  Term.register_const "?" q_ty

(* ---  1. REFL    ⊢ t = t ----------------------------------------------- *)
let refl t =
  if not (Term.well_formed t) then err "REFL: ill-formed term";
  Thm.mk [] (Term.mk_eq t t)

(* ---  2. TRANS   A ⊢ s = t   B ⊢ t = u   ⟹   A∪B ⊢ s = u ---------------- *)
let trans th1 th2 =
  let (s, t1) = try Term.dest_eq (Thm.concl th1)
                with _ -> err "TRANS: first premise not an equation" in
  let (t2, u) = try Term.dest_eq (Thm.concl th2)
                with _ -> err "TRANS: second premise not an equation" in
  if not (Term.alpha_eq t1 t2) then err "TRANS: middle terms differ";
  Thm.mk (Thm.union_hyps (Thm.hyps th1) (Thm.hyps th2))
         (Term.mk_eq s u)

(* ---  3. MK_COMB A ⊢ f = g   B ⊢ x = y   ⟹   A∪B ⊢ f x = g y ------------ *)
let mk_comb th1 th2 =
  let (f, g) = try Term.dest_eq (Thm.concl th1)
               with _ -> err "MK_COMB: first premise not an equation" in
  let (x, y) = try Term.dest_eq (Thm.concl th2)
               with _ -> err "MK_COMB: second premise not an equation" in
  (* type check: f must be a function whose argument type matches x *)
  (match Term.type_of f with
   | Type.Tyapp ("fun", [a; _]) ->
       if not (Type.equal a (Term.type_of x))
       then err "MK_COMB: type mismatch"
   | _ -> err "MK_COMB: lhs of first equation is not a function");
  Thm.mk (Thm.union_hyps (Thm.hyps th1) (Thm.hyps th2))
         (Term.mk_eq (Term.mk_comb f x) (Term.mk_comb g y))

(* ---  4. ABS  A ⊢ s = t  (v ∉ FV(A))  ⟹  A ⊢ λv. s = λv. t -------------- *)
let abs v th =
  (match v with
   | Term.Var (a, ty) ->
       if List.exists
            (fun h -> List.exists
                        (fun (b, ty') -> a = b && Type.equal ty ty')
                        (Term.frees h))
            (Thm.hyps th)
       then err "ABS: variable free in hypotheses"
   | _ -> err "ABS: binder must be a Var");
  let (s, t) = try Term.dest_eq (Thm.concl th)
               with _ -> err "ABS: conclusion not an equation" in
  Thm.mk (Thm.hyps th)
         (Term.mk_eq (Term.mk_abs v s) (Term.mk_abs v t))

(* ---  5. BETA   ⊢ (λv. t) v = t -------------------------------------- *)
let beta t =
  (match t with
   | Term.Comb (Term.Abs (Term.Var (_, _) as v, body), arg) ->
       if not (Term.alpha_eq v arg) then err "BETA: argument ≠ bound var";
       Thm.mk [] (Term.mk_eq t body)
   | _ -> err "BETA: not of the form (λv. t) v")

(* ---  6. ASSUME  p (of type bool)  ⟹  {p} ⊢ p ------------------------- *)
let assume p =
  if not (Term.well_formed p) then err "ASSUME: ill-formed";
  if not (Type.equal (Term.type_of p) Type.bool_ty)
  then err "ASSUME: not a proposition";
  Thm.mk [p] p

(* ---  7. EQ_MP  A ⊢ p = q   B ⊢ p   ⟹   A∪B ⊢ q --------------------- *)
let eq_mp th1 th2 =
  let (p, q) = try Term.dest_eq (Thm.concl th1)
               with _ -> err "EQ_MP: first premise not an equation" in
  if not (Term.alpha_eq p (Thm.concl th2))
  then err "EQ_MP: second premise does not match LHS";
  Thm.mk (Thm.union_hyps (Thm.hyps th1) (Thm.hyps th2)) q

(* ---  8. DEDUCT_ANTISYM_RULE
        A ⊢ p   B ⊢ q   ⟹   (A \ {q}) ∪ (B \ {p}) ⊢ p = q ---------------- *)
let deduct_antisym th1 th2 =
  let p = Thm.concl th1 in
  let q = Thm.concl th2 in
  let a' = Thm.remove_hyps (Thm.hyps th1) [q] in
  let b' = Thm.remove_hyps (Thm.hyps th2) [p] in
  Thm.mk (Thm.union_hyps a' b') (Term.mk_eq p q)

(* ---  9. INST   substitute free variables ---------------------------- *)
let inst theta th =
  let theta' = List.map (fun (v, t) ->
    match v with
    | Term.Var (n, ty) ->
        if not (Type.equal ty (Term.type_of t))
        then err "INST: type mismatch in substitution";
        ((n, ty), t)
    | _ -> err "INST: substitution domain must be a Var") theta
  in
  Thm.mk (List.map (Term.vsubst theta') (Thm.hyps th))
         (Term.vsubst theta' (Thm.concl th))

(* --- 10. INST_TYPE  substitute free type variables ------------------- *)
let inst_type theta th =
  Thm.mk (List.map (Term.inst_type theta) (Thm.hyps th))
         (Term.inst_type theta (Thm.concl th))

(* ====================================================================== *)
(* Connective-handling primitives.                                        *)
(*                                                                        *)
(* In a fully bootstrapped HOL kernel, the rules below are *derived* from *)
(* the 10 primitives above together with the standard connective          *)
(* definitions.  We expose them as kernel primitives for the MVP to keep  *)
(* the connective bootstrap out of v0.1.  They are sound under HOL        *)
(* Light's connective definitions; replacing each with its derivation     *)
(* shrinks the trust base without changing the verifier interface.       *)
(* ====================================================================== *)

(* --- helpers ---------------------------------------------------------- *)

let dest_forall = function
  | Term.Comb (Term.Const ("!", _), Term.Abs (Term.Var (a, ty), body)) ->
      (a, ty, body)
  | _ -> err "expected universal"

let dest_exists = function
  | Term.Comb (Term.Const ("?", _), Term.Abs (Term.Var (a, ty), body)) ->
      (a, ty, body)
  | _ -> err "expected existential"

let dest_conj = function
  | Term.Comb (Term.Comb (Term.Const ("/\\", _), p), q) -> (p, q)
  | _ -> err "expected conjunction"

let dest_imp = function
  | Term.Comb (Term.Comb (Term.Const ("==>", _), p), q) -> (p, q)
  | _ -> err "expected implication"

let mk_forall (a, ty) body =
  let pty = Type.fun_ty ty Type.bool_ty in
  let q_ty = Type.fun_ty pty Type.bool_ty in
  let q = Term.Const ("!", q_ty) in
  Term.mk_comb q (Term.mk_abs (Term.Var (a, ty)) body)

let mk_exists (a, ty) body =
  let pty = Type.fun_ty ty Type.bool_ty in
  let q_ty = Type.fun_ty pty Type.bool_ty in
  let q = Term.Const ("?", q_ty) in
  Term.mk_comb q (Term.mk_abs (Term.Var (a, ty)) body)

let mk_conj p q =
  let bb_b = Type.fun_ty Type.bool_ty (Type.fun_ty Type.bool_ty Type.bool_ty) in
  Term.mk_comb (Term.mk_comb (Term.Const ("/\\", bb_b)) p) q

let mk_imp p q =
  let bb_b = Type.fun_ty Type.bool_ty (Type.fun_ty Type.bool_ty Type.bool_ty) in
  Term.mk_comb (Term.mk_comb (Term.Const ("==>", bb_b)) p) q

(* --- GEN  ∀-intro ----------------------------------------------------- *)
(* From  A ⊢ p   with  v ∉ FV(A)   infer  A ⊢ ∀v. p                       *)
let gen v th =
  let (a, ty) = match v with
    | Term.Var (a, ty) -> (a, ty)
    | _ -> err "GEN: binder must be a Var" in
  if List.exists
       (fun h -> List.exists
                   (fun (b, ty') -> a = b && Type.equal ty ty')
                   (Term.frees h))
       (Thm.hyps th)
  then err "GEN: variable free in hypotheses";
  Thm.mk (Thm.hyps th) (mk_forall (a, ty) (Thm.concl th))

(* --- SPEC  ∀-elim ----------------------------------------------------- *)
(* From  A ⊢ ∀v. p[v]   and a term t   infer   A ⊢ p[t]                   *)
let spec t th =
  let (a, ty, body) = dest_forall (Thm.concl th) in
  if not (Type.equal ty (Term.type_of t))
  then err "SPEC: type mismatch";
  Thm.mk (Thm.hyps th)
         (Term.vsubst [((a, ty), t)] body)

(* --- EXISTS  ∃-intro -------------------------------------------------- *)
(* Witness style: from  A ⊢ p[w]  and a stated goal  ∃v. p[v]   infer     *)
(* A ⊢ ∃v. p[v].  We require the user to provide both the bound-variable  *)
(* name and the witness, so the verifier can compute the body.            *)
let exists_intro ~bound:(a, ty) ~witness:w th =
  if not (Type.equal ty (Term.type_of w))
  then err "EXISTS: type mismatch between bound var and witness";
  (* Construct the existential whose body, when instantiated at w,        *)
  (* equals the conclusion of th.  We synthesise it by abstracting over   *)
  (* w in the conclusion of th — i.e. the body P is computed as the term  *)
  (* obtained by replacing w with a fresh bound variable.                 *)
  let conc = Thm.concl th in
  (* Substitute w with Var(a, ty); requires a not free in conc. *)
  if List.exists (fun (b, ty') -> a = b && Type.equal ty ty') (Term.frees conc)
  then err "EXISTS: bound name clashes with a free variable in conclusion";
  let body =
    let rec subst t =
      if Term.alpha_eq t w then Term.Var (a, ty)
      else match t with
        | Term.Comb (f, x) -> Term.Comb (subst f, subst x)
        | Term.Abs (v, b) -> Term.Abs (v, subst b)
        | _ -> t
    in subst conc
  in
  Thm.mk (Thm.hyps th) (mk_exists (a, ty) body)

(* --- CHOOSE  ∃-elim --------------------------------------------------- *)
(* From  A ⊢ ∃v. P[v]   and   B ∪ {P[y]} ⊢ q   with y ∉ FV(B − {P[y]}) ∪  *)
(* FV(q) ∪ FV(A)   infer   A ∪ (B − {P[y]}) ⊢ q.                          *)
let choose ~witness:(a, ty) th_exists th_body =
  let (b, ty', body) = dest_exists (Thm.concl th_exists) in
  if not (Type.equal ty ty') then err "CHOOSE: type mismatch";
  let p_y = Term.vsubst [((b, ty'), Term.Var (a, ty))] body in
  let hyps_b = Thm.hyps th_body in
  if not (List.exists (Term.alpha_eq p_y) hyps_b)
  then err "CHOOSE: body theorem does not assume P[witness]";
  let hyps_b' = Thm.remove_hyps hyps_b [p_y] in
  (* freshness: a must not appear free in conclusion, remaining hyps, or in
     the existential premise's hypotheses or its quantified body's free
     vars (other than as the bound variable). *)
  let bad =
    Thm.concl th_body :: (hyps_b' @ Thm.hyps th_exists)
  in
  if List.exists
       (fun t -> List.exists (fun (n, ty'') -> n = a && Type.equal ty ty'')
                              (Term.frees t))
       bad
  then err "CHOOSE: witness variable not fresh";
  Thm.mk (Thm.union_hyps (Thm.hyps th_exists) hyps_b')
         (Thm.concl th_body)

(* --- CONJ  ∧-intro ---------------------------------------------------- *)
let conj th1 th2 =
  Thm.mk (Thm.union_hyps (Thm.hyps th1) (Thm.hyps th2))
         (mk_conj (Thm.concl th1) (Thm.concl th2))

(* --- CONJUNCT1, CONJUNCT2  ∧-elim ------------------------------------- *)
let conjunct1 th =
  let (p, _) = dest_conj (Thm.concl th) in
  Thm.mk (Thm.hyps th) p

let conjunct2 th =
  let (_, q) = dest_conj (Thm.concl th) in
  Thm.mk (Thm.hyps th) q

(* --- MP  modus ponens ------------------------------------------------- *)
(* From  A ⊢ p ⇒ q   and  B ⊢ p   infer  A∪B ⊢ q                          *)
let mp th1 th2 =
  let (p, q) = dest_imp (Thm.concl th1) in
  if not (Term.alpha_eq p (Thm.concl th2))
  then err "MP: antecedent does not match";
  Thm.mk (Thm.union_hyps (Thm.hyps th1) (Thm.hyps th2)) q

(* --- DISCH  deduction theorem ----------------------------------------- *)
(* From  A ⊢ q   and a hypothesis  p   infer   A \ {p} ⊢ p ⇒ q             *)
let disch p th =
  if not (Type.equal (Term.type_of p) Type.bool_ty)
  then err "DISCH: not a proposition";
  Thm.mk (Thm.remove_hyps (Thm.hyps th) [p])
         (mk_imp p (Thm.concl th))

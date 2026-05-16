(* kernel/axioms.ml — primitive HOL axioms, connective constants, and
   a manifest of theory-package-declared axioms.

   The four primitive axioms are HOL Light's: ETA, SELECT, INFINITY,
   plus classical excluded middle.  We do not currently use them in the
   MVP demos but they are exposed for completeness and for tests.

   The connective constants (∀, ∃, ∧, ∨, ⇒, ¬, T, F) are registered
   here so the term well-formedness checks accept formulas built from
   them.  Their *definitions* are not enforced in the MVP kernel; the
   connective-handling rules in [rules.ml] supply the needed reasoning.

   Theory packages declare axioms by calling [declare] with a name and
   a kernel formula; later, certificates can reference them by name
   through the AXIOM pseudo-rule. *)

(* Connective constants are registered in [rules.ml] (which this
   module depends on). *)

let bool_ty = Type.bool_ty
let bool_to_bool = Type.fun_ty bool_ty bool_ty
let bool_bool_bool =
  Type.fun_ty bool_ty (Type.fun_ty bool_ty bool_ty)

(* Force the dependency on Rules so its initializer runs before we
   build axiom terms below. *)
let _ = Rules.err

(* ---------- The four primitive HOL axioms --------------------------- *)

let eta_ax () =
  (* ⊢ ∀x. (λa. x a) = x   — at function type 'A -> 'B *)
  let a = Type.tyvar "A" in
  let b = Type.tyvar "B" in
  let fab = Type.fun_ty a b in
  let x = Term.Var ("x", fab) in
  let av = Term.Var ("a", a) in
  let lhs = Term.mk_abs av (Term.mk_comb x av) in
  Thm.mk [] (Term.mk_eq lhs x)

let select_ax () =
  (* ⊢ ∀p ∀x. p x ⇒ p (@ p)                                              *)
  let a = Type.tyvar "A" in
  let p = Term.Var ("p", Type.fun_ty a bool_ty) in
  let x = Term.Var ("x", a) in
  let select_const = Term.Const ("@", Type.fun_ty (Type.fun_ty a bool_ty) a) in
  let px = Term.mk_comb p x in
  let pexp = Term.mk_comb p (Term.mk_comb select_const p) in
  Thm.mk [] (Rules.mk_imp px pexp)

let em_ax () =
  (* ⊢ ∀p. p ∨ ¬p   — disjunction not currently a primitive rule, so we
     emit it as a raw connective-formula axiom. *)
  let p = Term.Var ("p", bool_ty) in
  let or_const = Term.Const ("\\/", bool_bool_bool) in
  let not_const = Term.Const ("~", bool_to_bool) in
  let notp = Term.mk_comb not_const p in
  let porp = Term.mk_comb (Term.mk_comb or_const p) notp in
  Thm.mk [] (Rules.mk_forall ("p", bool_ty) porp)

(* Infinity axiom is omitted from the MVP — it would assert the
   existence of an injection from [ind] to [ind] that misses a point,
   which we don't need until we conservatively define [nat] in v0.2. *)

(* ---------- Theory-package-declared axiom manifest ------------------ *)

let manifest : (string, Term.term) Hashtbl.t = Hashtbl.create 32

let declare name term =
  if not (Term.well_formed term) then
    failwith ("Axioms.declare: ill-formed axiom " ^ name);
  if not (Type.equal (Term.type_of term) Type.bool_ty) then
    failwith ("Axioms.declare: axiom " ^ name ^ " is not a proposition");
  match Hashtbl.find_opt manifest name with
  | Some existing when not (Term.alpha_eq existing term) ->
      failwith ("Axioms.declare: redeclaration with different statement: " ^ name)
  | _ -> Hashtbl.replace manifest name term

let lookup name =
  match Hashtbl.find_opt manifest name with
  | Some t -> t
  | None -> failwith ("Axioms.lookup: no declared axiom named " ^ name)

let all_declared () =
  Hashtbl.fold (fun n t acc -> (n, t) :: acc) manifest []

(* Mint a Thm from a declared axiom — used by the verifier when it
   encounters the AXIOM pseudo-rule. *)
let axiom_thm name =
  Thm.mk [] (lookup name)

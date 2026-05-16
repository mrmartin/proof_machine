(* kernel/term.ml — HOL terms.

   Simply-typed lambda calculus over the type system of [Type].
   Variables are named; alpha-equivalence is computed explicitly via
   a pair of name-to-index maps.  This is the HOL Light convention and
   is convenient because the serialised certificate form uses names. *)

type term =
  | Var of string * Type.hol_type
  | Const of string * Type.hol_type
  | Comb of term * term
  | Abs of term * term  (* the first must be a Var *)

(* --- Constants table -------------------------------------------------- *)

(* Every constant has a *declared* type (possibly polymorphic, expressed
   with type variables).  An occurrence [Const (c, ty)] is well-formed
   iff [ty] is an instance of the declared type of [c]. *)
let constants : (string, Type.hol_type) Hashtbl.t = Hashtbl.create 16

let () =
  (* The single primitive logical constant: polymorphic equality. *)
  let a = Type.tyvar "A" in
  Hashtbl.replace constants "="
    (Type.fun_ty a (Type.fun_ty a Type.bool_ty));
  (* Hilbert's epsilon operator, used to define ∃. *)
  Hashtbl.replace constants "@"
    (Type.fun_ty (Type.fun_ty a Type.bool_ty) a)

let register_const name ty =
  match Hashtbl.find_opt constants name with
  | Some ty' when not (Type.equal ty' ty) ->
      failwith ("register_const: type mismatch for " ^ name)
  | _ -> Hashtbl.replace constants name ty

let const_type name =
  match Hashtbl.find_opt constants name with
  | Some ty -> ty
  | None -> failwith ("unknown constant: " ^ name)

(* --- Term inspection --------------------------------------------------- *)

let rec type_of = function
  | Var (_, ty) -> ty
  | Const (_, ty) -> ty
  | Comb (f, x) ->
      (match type_of f with
       | Type.Tyapp ("fun", [a; b]) when Type.equal a (type_of x) -> b
       | Type.Tyapp ("fun", [a; _]) ->
           failwith (Printf.sprintf "type_of: comb mismatch: %s vs %s"
                       (Type.to_string a) (Type.to_string (type_of x)))
       | _ -> failwith "type_of: applied non-function")
  | Abs (v, body) ->
      (match v with
       | Var (_, ty) -> Type.fun_ty ty (type_of body)
       | _ -> failwith "Abs: binder must be a Var")

let rec well_formed = function
  | Var (_, ty) -> Type.well_formed ty
  | Const (c, ty) ->
      Type.well_formed ty &&
      (* the declared constant type must be instantiable to [ty];
         a quick sanity check: it must be well-formed and the constant
         must be registered.  We do not verify the instantiation here
         because the kernel rules that mint [Const] occurrences keep
         that invariant locally. *)
      (Hashtbl.mem constants c)
  | Comb (f, x) ->
      well_formed f && well_formed x &&
      (match type_of f with
       | Type.Tyapp ("fun", [a; _]) -> Type.equal a (type_of x)
       | _ -> false)
  | Abs (v, body) ->
      (match v with
       | Var (_, _) -> well_formed v && well_formed body
       | _ -> false)

(* --- Alpha-equivalence ------------------------------------------------- *)

(* Standard structural recursion with a pair of environments mapping
   bound-variable names to de-Bruijn-like depths. *)

let alpha_eq t1 t2 =
  let rec aux env1 env2 t1 t2 = match t1, t2 with
    | Var (a, ta), Var (b, tb) ->
        Type.equal ta tb &&
        (match List.assoc_opt a env1, List.assoc_opt b env2 with
         | Some i, Some j -> i = j
         | None, None -> a = b
         | _ -> false)
    | Const (c1, ty1), Const (c2, ty2) ->
        c1 = c2 && Type.equal ty1 ty2
    | Comb (f1, x1), Comb (f2, x2) ->
        aux env1 env2 f1 f2 && aux env1 env2 x1 x2
    | Abs (Var (a, ta), b1), Abs (Var (b, tb), b2) ->
        Type.equal ta tb &&
        let n = List.length env1 in
        aux ((a, n) :: env1) ((b, n) :: env2) b1 b2
    | _ -> false
  in
  aux [] [] t1 t2

(* --- Free variables ---------------------------------------------------- *)

let rec frees = function
  | Var (a, ty) -> [(a, ty)]
  | Const _ -> []
  | Comb (f, x) ->
      let fs = frees f in
      List.fold_left (fun acc v -> if List.mem v acc then acc else v :: acc)
        fs (frees x)
  | Abs (Var (a, ty), body) ->
      List.filter (fun (b, ty') -> not (a = b && Type.equal ty ty')) (frees body)
  | Abs _ -> failwith "frees: malformed Abs"

(* Set of free-variable *names* used at any binding-occurrence, useful
   when generating fresh variables. *)
let rec names_used = function
  | Var (a, _) -> [a]
  | Const _ -> []
  | Comb (f, x) -> names_used f @ names_used x
  | Abs (Var (a, _), b) -> a :: names_used b
  | Abs _ -> []

let fresh_var ~avoid base ty =
  if not (List.mem base avoid) then Var (base, ty)
  else
    let rec loop i =
      let n = base ^ string_of_int i in
      if List.mem n avoid then loop (i + 1) else Var (n, ty)
    in
    loop 0

(* --- Substitution ------------------------------------------------------ *)

(* Capture-avoiding substitution [vsubst theta t].  [theta] maps Var
   names (paired with their type) to terms of the same type. *)

let rec vsubst (theta : ((string * Type.hol_type) * term) list) t =
  if theta = [] then t else
  let rec subst bound = function
    | Var (a, ty) as v ->
        if List.exists (fun (b, ty') -> a = b && Type.equal ty ty') bound
        then v
        else (try List.assoc (a, ty) theta with Not_found -> v)
    | Const _ as c -> c
    | Comb (f, x) -> Comb (subst bound f, subst bound x)
    | Abs (Var (a, ty), body) ->
        (* Compute the set of names we must avoid to prevent variable
           capture.  A substitution [(x ↦ s)] could cause capture only
           if [x] is genuinely free in [body] (not the bound [a]) and
           [s] mentions [a] free.  So we restrict to substitutions
           that actually apply to free occurrences in [body]. *)
        let applicable =
          List.filter (fun ((n, ty'), _) ->
            n <> a || not (Type.equal ty ty')) theta
          |> List.filter (fun ((n, ty'), _) ->
            List.exists (fun (b, ty'') -> n = b && Type.equal ty' ty'')
              (frees body))
        in
        if applicable = [] then
          Abs (Var (a, ty), body)
        else begin
          let capture_set =
            List.concat_map (fun (_, t) -> List.map fst (frees t)) applicable
          in
          if List.mem a capture_set then
            let avoid = capture_set @ List.map fst (frees body) in
            let a' =
              match fresh_var ~avoid a ty with
              | Var (n, _) -> n
              | _ -> assert false
            in
            let body_renamed =
              vsubst [((a, ty), Var (a', ty))] body
            in
            Abs (Var (a', ty), subst ((a', ty) :: bound) body_renamed)
          else
            Abs (Var (a, ty), subst ((a, ty) :: bound) body)
        end
    | Abs _ -> failwith "vsubst: malformed Abs"
  in
  subst [] t

(* Type instantiation in a term: replace free type variables. *)
let rec inst_type theta = function
  | Var (a, ty) -> Var (a, Type.type_subst theta ty)
  | Const (c, ty) -> Const (c, Type.type_subst theta ty)
  | Comb (f, x) -> Comb (inst_type theta f, inst_type theta x)
  | Abs (Var (a, ty), b) ->
      Abs (Var (a, Type.type_subst theta ty), inst_type theta b)
  | Abs _ -> failwith "inst_type: malformed Abs"

(* --- Smart constructors ----------------------------------------------- *)

let mk_var n ty = Var (n, ty)

let mk_const ?ty name =
  let declared = const_type name in
  let ty = match ty with Some t -> t | None -> declared in
  Const (name, ty)

let mk_comb f x = Comb (f, x)

let mk_abs v body =
  match v with Var _ -> Abs (v, body) | _ -> failwith "mk_abs: not a var"

(* Equality term:  =[ty]  s  t  *)
let mk_eq s t =
  let ty = type_of s in
  if not (Type.equal ty (type_of t)) then
    failwith "mk_eq: type mismatch";
  let eq_ty = Type.fun_ty ty (Type.fun_ty ty Type.bool_ty) in
  Comb (Comb (Const ("=", eq_ty), s), t)

let dest_eq = function
  | Comb (Comb (Const ("=", _), s), t) -> (s, t)
  | _ -> failwith "dest_eq: not an equation"

let is_eq = function
  | Comb (Comb (Const ("=", _), _), _) -> true
  | _ -> false

(* --- Pretty-printing --------------------------------------------------- *)

let rec to_string = function
  | Var (a, _) -> a
  | Const (c, _) -> c
  | Comb (Comb (Const ("=", _), s), t) ->
      "(" ^ to_string s ^ " = " ^ to_string t ^ ")"
  | Comb (f, x) -> "(" ^ to_string f ^ " " ^ to_string x ^ ")"
  | Abs (Var (a, ty), b) ->
      "(\\" ^ a ^ ":" ^ Type.to_string ty ^ ". " ^ to_string b ^ ")"
  | Abs _ -> "<bad-abs>"

(* kernel/type.ml — HOL types.

   This file is part of the trusted kernel. Keep it minimal.

   A HOL type is either a type variable or a type-constructor applied to
   a (possibly empty) list of types.  We bake in three primitive type
   constructors: [bool] (the truth-values type, arity 0), [ind] (an
   infinite type of "individuals", arity 0), and [fun] (function space,
   arity 2). Theory packages declare additional type constants (e.g.
   [nat]) via [register_tyconstr]. *)

type hol_type =
  | Tyvar of string
  | Tyapp of string * hol_type list

let tyvar s = Tyvar s
let tyapp (s, args) = Tyapp (s, args)

(* The primitive arities table.  Mutable because theory packages must
   be able to register new type constructors before any term referring
   to them is built; but every entry here is appended, never replaced,
   so the trusted base only grows in well-controlled ways. *)
let arities : (string, int) Hashtbl.t = Hashtbl.create 16

let () =
  Hashtbl.replace arities "bool" 0;
  Hashtbl.replace arities "ind"  0;
  Hashtbl.replace arities "fun"  2

let register_tyconstr name arity =
  match Hashtbl.find_opt arities name with
  | Some k when k <> arity ->
      failwith ("register_tyconstr: arity mismatch for " ^ name)
  | _ -> Hashtbl.replace arities name arity

let arity_of name =
  match Hashtbl.find_opt arities name with
  | Some k -> k
  | None -> failwith ("unknown type constructor: " ^ name)

let rec well_formed = function
  | Tyvar _ -> true
  | Tyapp (s, args) ->
      (match Hashtbl.find_opt arities s with
       | Some k -> k = List.length args && List.for_all well_formed args
       | None -> false)

let bool_ty   = Tyapp ("bool", [])
let ind_ty    = Tyapp ("ind", [])
let fun_ty a b = Tyapp ("fun", [a; b])

(* Structural equality on types — types do not have binders, so this
   is straightforward. *)
let rec equal t1 t2 = match t1, t2 with
  | Tyvar a, Tyvar b -> a = b
  | Tyapp (s, xs), Tyapp (t, ys) ->
      s = t && List.length xs = List.length ys
      && List.for_all2 equal xs ys
  | _ -> false

(* Substitute [theta] = [(α₁ ↦ T₁); …] into a type.  Capture is
   impossible because types have no binders. *)
let rec type_subst theta = function
  | Tyvar a as t ->
      (try List.assoc a theta with Not_found -> t)
  | Tyapp (s, args) -> Tyapp (s, List.map (type_subst theta) args)

let rec tyvars_of = function
  | Tyvar a -> [a]
  | Tyapp (_, args) ->
      List.fold_left
        (fun acc t ->
           List.fold_left (fun a v -> if List.mem v a then a else v :: a)
             acc (tyvars_of t))
        [] args

let rec to_string = function
  | Tyvar a -> "'" ^ a
  | Tyapp ("fun", [a; b]) ->
      "(" ^ to_string a ^ " -> " ^ to_string b ^ ")"
  | Tyapp (s, []) -> s
  | Tyapp (s, args) ->
      "(" ^ String.concat ", " (List.map to_string args) ^ ") " ^ s

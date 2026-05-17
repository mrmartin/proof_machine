(* tokenizer/encode.ml — Cert.t / Term.term → token-ID stream.

   Walks the kernel AST, allocating pool slots for theory-scoped names
   (constants, axiom names, declared type constructors, type variables)
   and bound variables on first sight.  The result is an [int list]
   (token IDs in [0 .. Lexicon.vocab_size - 1]) plus a [pool_header]
   that records the slot → source-name map.

   The grammar of the token stream is exactly what [Decode] consumes
   and what [Grammar] accepts. *)

open Kernel
module L = Lexicon

type pool_header = {
  tycons : string array;
  tyvars : string array;
  names  : string array;
  vars   : string array;   (* source-name per allocated var-pool slot *)
}

exception Encode_error of string
let err s = raise (Encode_error s)

type ctx = {
  mutable tycons : string list;  (* reverse alloc order *)
  mutable tyvars : string list;
  mutable names  : string list;
  mutable vars   : string list;
  mutable out    : int list;
}

let make_ctx () =
  { tycons = []; tyvars = []; names = []; vars = []; out = [] }

let emit ctx tok = ctx.out <- tok :: ctx.out

let alloc lst limit kind name =
  let rec find i = function
    | [] -> None
    | x :: rest -> if x = name then Some i else find (i + 1) rest
  in
  match find 0 (List.rev lst) with
  | Some i -> (i, lst)
  | None ->
    let n = List.length lst in
    if n >= limit then err (Printf.sprintf "%s pool full (limit %d) at %s"
                              kind limit name);
    (n, name :: lst)

let alloc_tycon ctx name =
  let i, lst' = alloc ctx.tycons L.tycon_count "tycon" name in
  ctx.tycons <- lst';
  i

let alloc_tyvar ctx name =
  let i, lst' = alloc ctx.tyvars L.tyvar_count "tyvar" name in
  ctx.tyvars <- lst';
  i

let alloc_name ctx name =
  let i, lst' = alloc ctx.names L.name_count "name" name in
  ctx.names <- lst';
  i

let alloc_var ctx name =
  let i, lst' = alloc ctx.vars L.var_count "var" name in
  ctx.vars <- lst';
  i

(* --- Type encoder ---------------------------------------------------- *)

let rec encode_type ctx ty =
  match ty with
  | Type.Tyvar a ->
    let k = alloc_tyvar ctx a in
    emit ctx (L.tyvar_tok k)
  | Type.Tyapp ("fun", [a; b]) ->
    (* render as (a -> b) *)
    emit ctx L.lparen;
    encode_type ctx a;
    emit ctx L.arrow;
    encode_type ctx b;
    emit ctx L.rparen
  | Type.Tyapp (name, []) ->
    (match L.builtin_of_name name with
     | Some t -> emit ctx t
     | None -> let k = alloc_tycon ctx name in emit ctx (L.tycon_tok k))
  | Type.Tyapp (name, args) ->
    let head_tok = match L.builtin_of_name name with
      | Some t -> t
      | None -> L.tycon_tok (alloc_tycon ctx name)
    in
    emit ctx L.lparen;
    let rec loop = function
      | [] -> ()
      | [t] -> encode_type ctx t
      | t :: rest -> encode_type ctx t; emit ctx L.comma; loop rest
    in
    loop args;
    emit ctx L.rparen;
    emit ctx head_tok

(* --- Term encoder ---------------------------------------------------- *)

let encode_int ctx n =
  if n < 0 || n >= L.int_count then
    err (Printf.sprintf "integer literal %d out of range" n);
  emit ctx (L.int_tok n)

let encode_var_ref ctx name =
  let k = alloc_var ctx name in
  emit ctx (L.var_tok k)

let encode_const_ref ctx name =
  (* Reserved-symbol constants (=, /\, \/, ==>, ~, !, ?) are rendered as
     their operator tokens at the relevant Comb-pattern site; bare
     occurrences (rare) fall through here. *)
  match name with
  | "=" -> emit ctx L.op_eq
  | "/\\" -> emit ctx L.op_conj
  | "\\/" -> emit ctx L.op_disj
  | "==>" -> emit ctx L.op_imp
  | "~" -> emit ctx L.op_not
  | "!" -> emit ctx L.op_forall
  | "?" -> emit ctx L.op_exists
  | _ ->
    let k = alloc_name ctx name in
    emit ctx (L.name_tok k)

let rec encode_term ctx t =
  match t with
  | Term.Comb (Term.Comb (Term.Const ("=", _), a), b) ->
    emit ctx L.lparen; encode_term ctx a;
    emit ctx L.op_eq;
    encode_term ctx b; emit ctx L.rparen
  | Term.Comb (Term.Comb (Term.Const ("/\\", _), a), b) ->
    emit ctx L.lparen; encode_term ctx a;
    emit ctx L.op_conj;
    encode_term ctx b; emit ctx L.rparen
  | Term.Comb (Term.Comb (Term.Const ("\\/", _), a), b) ->
    emit ctx L.lparen; encode_term ctx a;
    emit ctx L.op_disj;
    encode_term ctx b; emit ctx L.rparen
  | Term.Comb (Term.Comb (Term.Const ("==>", _), a), b) ->
    emit ctx L.lparen; encode_term ctx a;
    emit ctx L.op_imp;
    encode_term ctx b; emit ctx L.rparen
  | Term.Comb (Term.Const ("~", _), b) ->
    emit ctx L.lparen; emit ctx L.op_not; encode_term ctx b; emit ctx L.rparen
  | Term.Comb (Term.Const ("!", _), Term.Abs (Term.Var (n, ty), body)) ->
    encode_binder ctx L.op_forall n ty body
  | Term.Comb (Term.Const ("?", _), Term.Abs (Term.Var (n, ty), body)) ->
    encode_binder ctx L.op_exists n ty body
  | Term.Abs (Term.Var (n, ty), body) ->
    encode_binder ctx L.op_lambda n ty body
  | Term.Comb (f, x) ->
    emit ctx L.lparen; encode_term ctx f; encode_term ctx x; emit ctx L.rparen
  | Term.Var (n, ty) ->
    encode_var_ref ctx n;
    emit ctx L.colon;
    encode_type ctx ty
  | Term.Const (c, _) ->
    encode_const_ref ctx c
  | Term.Abs _ -> err "Encode: malformed Abs"

and encode_binder ctx op n ty body =
  emit ctx L.lparen;
  emit ctx op;
  encode_var_ref ctx n;
  emit ctx L.colon;
  encode_type ctx ty;
  emit ctx L.dot;
  encode_term ctx body;
  emit ctx L.rparen

(* --- Witness encoder ------------------------------------------------- *)

let encode_witness ctx w =
  match w with
  | Cert.W_none -> ()
  | Cert.W_term t ->
    emit ctx L.lparen; emit ctx L.kw_term;
    emit ctx L.quote; encode_term ctx t; emit ctx L.quote;
    emit ctx L.rparen
  | Cert.W_type ty ->
    emit ctx L.lparen; emit ctx L.kw_type;
    emit ctx L.quote; encode_type ctx ty; emit ctx L.quote;
    emit ctx L.rparen
  | Cert.W_var (n, ty) ->
    emit ctx L.lparen; emit ctx L.kw_var;
    emit ctx L.quote; encode_var_ref ctx n; emit ctx L.quote;
    emit ctx L.quote; encode_type ctx ty; emit ctx L.quote;
    emit ctx L.rparen
  | Cert.W_axiom name ->
    emit ctx L.lparen; emit ctx L.kw_axiom;
    emit ctx L.quote;
    let k = alloc_name ctx name in
    emit ctx (L.name_tok k);
    emit ctx L.quote;
    emit ctx L.rparen
  | Cert.W_inst pairs ->
    emit ctx L.lparen; emit ctx L.kw_inst;
    List.iter (fun (v, rhs) ->
      emit ctx L.lparen; emit ctx L.kw_subst;
      emit ctx L.quote; encode_term ctx v; emit ctx L.quote;
      emit ctx L.quote; encode_term ctx rhs; emit ctx L.quote;
      emit ctx L.rparen) pairs;
    emit ctx L.rparen
  | Cert.W_inst_type pairs ->
    emit ctx L.lparen; emit ctx L.kw_insttype;
    List.iter (fun (a, ty) ->
      emit ctx L.lparen; emit ctx L.kw_subst;
      emit ctx L.quote; encode_var_ref ctx a; emit ctx L.quote;
      emit ctx L.quote; encode_type ctx ty; emit ctx L.quote;
      emit ctx L.rparen) pairs;
    emit ctx L.rparen
  | Cert.W_bound_and_witness ((n, ty), w_term) ->
    emit ctx L.lparen; emit ctx L.kw_b_and_w;
    emit ctx L.lparen; emit ctx L.kw_bound;
    emit ctx L.quote; encode_var_ref ctx n; emit ctx L.quote;
    emit ctx L.quote; encode_type ctx ty; emit ctx L.quote;
    emit ctx L.rparen;
    emit ctx L.lparen; emit ctx L.kw_witness;
    emit ctx L.quote; encode_term ctx w_term; emit ctx L.quote;
    emit ctx L.rparen;
    emit ctx L.rparen

(* --- Step encoder ---------------------------------------------------- *)

let encode_step ctx (s : Cert.step) =
  emit ctx L.lparen; emit ctx L.kw_step;
  encode_int ctx s.id;
  emit ctx L.lparen; emit ctx L.kw_rule;
  (match L.rule_of_name s.rule with
   | Some r -> emit ctx r
   | None -> err ("unknown rule name: " ^ s.rule));
  emit ctx L.rparen;
  emit ctx L.lparen; emit ctx L.kw_witness;
  encode_witness ctx s.witness;
  emit ctx L.rparen;
  emit ctx L.lparen; emit ctx L.kw_premises;
  List.iter (encode_int ctx) s.premises;
  emit ctx L.rparen;
  emit ctx L.rparen

(* --- Cert encoder ---------------------------------------------------- *)

let cert (c : Cert.t) : int array * pool_header =
  let ctx = make_ctx () in
  emit ctx L.lparen; emit ctx L.kw_cert;
  List.iter (encode_step ctx) c.steps;
  emit ctx L.lparen; emit ctx L.kw_concl;
  emit ctx L.quote; encode_term ctx c.concl; emit ctx L.quote;
  emit ctx L.rparen;
  emit ctx L.rparen;
  let toks = Array.of_list (List.rev ctx.out) in
  let hdr = {
    tycons = Array.of_list (List.rev ctx.tycons);
    tyvars = Array.of_list (List.rev ctx.tyvars);
    names  = Array.of_list (List.rev ctx.names);
    vars   = Array.of_list (List.rev ctx.vars);
  } in
  (toks, hdr)

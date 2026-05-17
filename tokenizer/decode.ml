(* tokenizer/decode.ml — token-ID stream → Cert.t.

   Inverse of [Encode].  Returns a [Cert.t] that is alpha-equivalent to
   the original; variable *names* are canonicalised to ["v0", "v1", …]
   (the pool slot), while constant and axiom names are recovered from
   the [pool_header] passed in by the encoder.

   The decoder is a straightforward recursive descent — no backtracking,
   no lookahead beyond one token — because the grammar is unambiguous
   by construction (the encoder writes fully-parenthesised forms). *)

open Kernel
module L = Lexicon

type pool_header = Encode.pool_header

exception Decode_error of string
let err s = raise (Decode_error s)

type cursor = {
  toks : int array;
  mutable pos : int;
}

let peek c =
  if c.pos >= Array.length c.toks then err "decode: unexpected end of stream"
  else c.toks.(c.pos)

let peek_opt c =
  if c.pos >= Array.length c.toks then None else Some c.toks.(c.pos)

let advance c = c.pos <- c.pos + 1

let expect c tok =
  if peek c <> tok then
    err (Printf.sprintf "decode: expected %s, got %s at pos %d"
           (L.to_string tok) (L.to_string (peek c)) c.pos);
  advance c

let pop_int c =
  match L.int_of_int_tok (peek c) with
  | Some n -> advance c; n
  | None -> err (Printf.sprintf "decode: expected INT, got %s at pos %d"
                   (L.to_string (peek c)) c.pos)

(* --- Type decoder ---------------------------------------------------- *)

let lookup_tycon (hdr : pool_header) k =
  if k < Array.length hdr.tycons then hdr.tycons.(k)
  else err (Printf.sprintf "decode: tycon slot %d out of bounds" k)

let lookup_tyvar (hdr : pool_header) k =
  if k < Array.length hdr.tyvars then hdr.tyvars.(k)
  else err (Printf.sprintf "decode: tyvar slot %d out of bounds" k)

let lookup_name (hdr : pool_header) k =
  if k < Array.length hdr.names then hdr.names.(k)
  else err (Printf.sprintf "decode: name slot %d out of bounds" k)

let rec decode_type hdr c =
  let tok = peek c in
  if tok = L.lparen then begin
    advance c;
    (* Two cases:
         "( arg1 -> arg2 )"   --- function type (arrow appears at top)
         "( arg1 , arg2 ) Tycon"  --- type application with multiple args *)
    let first = decode_type hdr c in
    let nxt = peek c in
    if nxt = L.arrow then begin
      advance c;
      let rhs = decode_type hdr c in
      expect c L.rparen;
      Type.fun_ty first rhs
    end else begin
      let rec loop acc =
        match peek c with
        | t when t = L.comma -> advance c; loop (decode_type hdr c :: acc)
        | t when t = L.rparen -> advance c; List.rev acc
        | t -> err (Printf.sprintf "decode_type: unexpected %s" (L.to_string t))
      in
      let args = loop [first] in
      let head = peek c in
      let name =
        if head = L.ty_bool || head = L.ty_ind || head = L.ty_fun || head = L.ty_nat then
          (match L.name_of_builtin head with
           | Some n -> advance c; n
           | None -> err "decode_type: builtin")
        else match L.tycon_index head with
          | Some k -> advance c; lookup_tycon hdr k
          | None -> err (Printf.sprintf "decode_type: expected tycon head, got %s"
                           (L.to_string head))
      in
      Type.Tyapp (name, args)
    end
  end
  else if L.is_tyvar tok then begin
    advance c;
    match L.tyvar_index tok with
    | Some k -> Type.Tyvar (lookup_tyvar hdr k)
    | None -> err "tyvar"
  end
  else match L.name_of_builtin tok with
    | Some n -> advance c; Type.Tyapp (n, [])
    | None ->
      match L.tycon_index tok with
      | Some k -> advance c; Type.Tyapp (lookup_tycon hdr k, [])
      | None -> err (Printf.sprintf "decode_type: unexpected %s" (L.to_string tok))

(* --- Term decoder ---------------------------------------------------- *)

let var_name_in (hdr : pool_header) k =
  if k < Array.length hdr.vars then hdr.vars.(k)
  else Printf.sprintf "v%d" k

let rec decode_term hdr c =
  let tok = peek c in
  if tok = L.lparen then begin
    advance c;
    let nxt = peek c in
    if nxt = L.op_forall then (advance c; decode_binder hdr c `Forall)
    else if nxt = L.op_exists then (advance c; decode_binder hdr c `Exists)
    else if nxt = L.op_lambda then (advance c; decode_binder hdr c `Lambda)
    else if nxt = L.op_not then begin
      advance c;
      let body = decode_term hdr c in
      expect c L.rparen;
      let not_const = Term.Const ("~", Type.fun_ty Type.bool_ty Type.bool_ty) in
      Term.mk_comb not_const body
    end
    else begin
      (* Either a binary operator form or an application form.  Parse
         one term, then peek at the next token. *)
      let t1 = decode_term hdr c in
      let op = peek c in
      if op = L.op_eq then begin
        advance c;
        let t2 = decode_term hdr c in
        expect c L.rparen;
        Term.mk_eq t1 t2
      end
      else if op = L.op_conj then begin
        advance c;
        let t2 = decode_term hdr c in
        expect c L.rparen;
        Rules.mk_conj t1 t2
      end
      else if op = L.op_disj then begin
        advance c;
        let t2 = decode_term hdr c in
        expect c L.rparen;
        let bb = Type.fun_ty Type.bool_ty (Type.fun_ty Type.bool_ty Type.bool_ty) in
        let or_const = Term.Const ("\\/", bb) in
        Term.mk_comb (Term.mk_comb or_const t1) t2
      end
      else if op = L.op_imp then begin
        advance c;
        let t2 = decode_term hdr c in
        expect c L.rparen;
        Rules.mk_imp t1 t2
      end
      else begin
        (* Application: gather successive terms until ')'. *)
        let rec gather acc =
          if peek c = L.rparen then (advance c; acc)
          else
            let nxt = decode_term hdr c in
            gather (Term.mk_comb acc nxt)
        in
        gather t1
      end
    end
  end
  else if L.is_var tok then begin
    advance c;
    let k = match L.var_index tok with
      | Some k -> k | None -> err "var" in
    expect c L.colon;
    let ty = decode_type hdr c in
    Term.Var (var_name_in hdr k, ty)
  end
  else if tok = L.op_eq then begin
    advance c;
    let a = Type.tyvar "A" in
    Term.Const ("=", Type.fun_ty a (Type.fun_ty a Type.bool_ty))
  end
  else if L.is_name tok then begin
    advance c;
    let k = match L.name_index tok with Some k -> k | None -> err "name" in
    let name = lookup_name hdr k in
    let ty = match Hashtbl.find_opt Term.constants name with
      | Some ty -> ty
      | None -> err ("decode_term: unknown constant: " ^ name)
    in
    Term.Const (name, ty)
  end
  else err (Printf.sprintf "decode_term: unexpected %s at pos %d"
              (L.to_string tok) c.pos)

and decode_binder hdr c kind =
  let var_tok = peek c in
  let k = match L.var_index var_tok with
    | Some k -> k | None -> err "binder: expected var" in
  advance c;
  expect c L.colon;
  let ty = decode_type hdr c in
  expect c L.dot;
  let body = decode_term hdr c in
  expect c L.rparen;
  let v = Term.Var (var_name_in hdr k, ty) in
  match kind with
  | `Forall -> Rules.mk_forall (var_name_in hdr k, ty) body
  | `Exists -> Rules.mk_exists (var_name_in hdr k, ty) body
  | `Lambda -> Term.mk_abs v body

(* --- Witness decoder ------------------------------------------------- *)

let decode_witness hdr c =
  (* Caller has already consumed `( KW_witness`.  We now read either:
       )                       — W_none
       ( <tag> ... ) )         — one of the tagged witnesses
     Returns the Cert.witness and leaves cursor positioned after the
     outer `)`. *)
  if peek c = L.rparen then (advance c; Cert.W_none)
  else begin
    expect c L.lparen;
    let tag = peek c in
    advance c;
    let w =
      if tag = L.kw_term then begin
        expect c L.quote;
        let t = decode_term hdr c in
        expect c L.quote;
        expect c L.rparen;
        Cert.W_term t
      end
      else if tag = L.kw_type then begin
        expect c L.quote;
        let ty = decode_type hdr c in
        expect c L.quote;
        expect c L.rparen;
        Cert.W_type ty
      end
      else if tag = L.kw_var then begin
        expect c L.quote;
        let var_tok = peek c in
        advance c;
        let k = match L.var_index var_tok with
          | Some k -> k | None -> err "W_var: expected var" in
        expect c L.quote;
        expect c L.quote;
        let ty = decode_type hdr c in
        expect c L.quote;
        expect c L.rparen;
        Cert.W_var (var_name_in hdr k, ty)
      end
      else if tag = L.kw_axiom then begin
        expect c L.quote;
        let nm_tok = peek c in
        advance c;
        let k = match L.name_index nm_tok with
          | Some k -> k | None -> err "W_axiom: expected name" in
        expect c L.quote;
        expect c L.rparen;
        Cert.W_axiom (lookup_name hdr k)
      end
      else if tag = L.kw_inst then begin
        let rec loop acc =
          if peek c = L.rparen then (advance c; List.rev acc)
          else begin
            expect c L.lparen;
            expect c L.kw_subst;
            expect c L.quote;
            let v = decode_term hdr c in
            expect c L.quote;
            expect c L.quote;
            let t = decode_term hdr c in
            expect c L.quote;
            expect c L.rparen;
            loop ((v, t) :: acc)
          end
        in
        Cert.W_inst (loop [])
      end
      else if tag = L.kw_insttype then begin
        let rec loop acc =
          if peek c = L.rparen then (advance c; List.rev acc)
          else begin
            expect c L.lparen;
            expect c L.kw_subst;
            expect c L.quote;
            let var_tok = peek c in
            advance c;
            let k = match L.var_index var_tok with
              | Some k -> k | None -> err "W_insttype: expected var" in
            expect c L.quote;
            expect c L.quote;
            let ty = decode_type hdr c in
            expect c L.quote;
            expect c L.rparen;
            loop ((var_name_in hdr k, ty) :: acc)
          end
        in
        Cert.W_inst_type (loop [])
      end
      else if tag = L.kw_b_and_w then begin
        expect c L.lparen;
        expect c L.kw_bound;
        expect c L.quote;
        let var_tok = peek c in
        advance c;
        let k = match L.var_index var_tok with
          | Some k -> k | None -> err "W_b_and_w: expected var" in
        expect c L.quote;
        expect c L.quote;
        let ty = decode_type hdr c in
        expect c L.quote;
        expect c L.rparen;
        expect c L.lparen;
        expect c L.kw_witness;
        expect c L.quote;
        let w_term = decode_term hdr c in
        expect c L.quote;
        expect c L.rparen;
        expect c L.rparen;
        Cert.W_bound_and_witness ((var_name_in hdr k, ty), w_term)
      end
      else err (Printf.sprintf "decode_witness: unknown tag %s" (L.to_string tag))
    in
    expect c L.rparen;  (* close outer (witness ...) *)
    w
  end

(* --- Step decoder ---------------------------------------------------- *)

let decode_step hdr c =
  expect c L.lparen;
  expect c L.kw_step;
  let id = pop_int c in
  expect c L.lparen;
  expect c L.kw_rule;
  let rule_tok = peek c in
  let rule_name = match L.name_of_rule rule_tok with
    | Some n -> advance c; n
    | None -> err (Printf.sprintf "decode_step: expected rule name, got %s"
                     (L.to_string rule_tok))
  in
  expect c L.rparen;
  expect c L.lparen;
  expect c L.kw_witness;
  let witness = decode_witness hdr c in
  expect c L.lparen;
  expect c L.kw_premises;
  let rec loop acc =
    if peek c = L.rparen then (advance c; List.rev acc)
    else loop (pop_int c :: acc)
  in
  let premises = loop [] in
  expect c L.rparen;
  { Cert.id; rule = rule_name; witness; premises; declared_concl = None }

(* --- Cert decoder ---------------------------------------------------- *)

let cert (hdr : pool_header) (toks : int array) : Cert.t =
  let c = { toks; pos = 0 } in
  expect c L.lparen;
  expect c L.kw_cert;
  let steps = ref [] in
  (* Steps continue until we see `( KW_concl ...`. *)
  let rec read_steps () =
    if peek c <> L.lparen then err "expected ( for step or concl";
    if c.pos + 1 < Array.length c.toks
       && c.toks.(c.pos + 1) = L.kw_concl then ()
    else begin
      let s = decode_step hdr c in
      steps := s :: !steps;
      read_steps ()
    end
  in
  read_steps ();
  expect c L.lparen;
  expect c L.kw_concl;
  expect c L.quote;
  let concl = decode_term hdr c in
  expect c L.quote;
  expect c L.rparen;
  expect c L.rparen;
  { Cert.steps = List.rev !steps; concl }

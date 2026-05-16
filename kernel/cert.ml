(* kernel/cert.ml — certificate format and rule dispatcher.

   A certificate is a sequence of [step]s plus a declared final
   conclusion.  Each step says: take these previously-checked premises,
   apply this rule with this witness, and you'll get this theorem.

   The S-expression syntax is documented in docs/CERTIFICATE.md. *)

(* ---------- Witness ADT ----------------------------------------------- *)

type witness =
  | W_none
  | W_term of Term.term
  | W_type of Type.hol_type
  | W_var of string * Type.hol_type
  | W_inst of (Term.term * Term.term) list
  | W_inst_type of (string * Type.hol_type) list
  | W_axiom of string
  | W_bound_and_witness of (string * Type.hol_type) * Term.term

(* ---------- Steps ----------------------------------------------------- *)

type step = {
  id : int;
  rule : string;
  witness : witness;
  premises : int list;
  declared_concl : Term.term option;
}

type t = {
  steps : step list;
  concl : Term.term;
}

(* ---------- Tiny S-expression parser --------------------------------- *)

(* The certificate is read as a single S-expression of the form
   (cert (step ...) (step ...) ... (concl "...")).  Quoted strings hold
   HOL terms/types written in the surface syntax accepted by Sexp_term
   below — which is a *small* subset just sufficient for the MVP. *)

type sexp = Atom of string | List of sexp list

let read_sexp s =
  let pos = ref 0 in
  let len = String.length s in
  let skip_ws () =
    while !pos < len && (let c = s.[!pos] in
                         c = ' ' || c = '\t' || c = '\n' || c = '\r'
                         || c = ';') do
      if s.[!pos] = ';' then
        while !pos < len && s.[!pos] <> '\n' do incr pos done
      else incr pos
    done
  in
  let rec parse_one () =
    skip_ws ();
    if !pos >= len then failwith "cert: unexpected EOF";
    match s.[!pos] with
    | '(' ->
        incr pos;
        let xs = ref [] in
        skip_ws ();
        while !pos < len && s.[!pos] <> ')' do
          xs := parse_one () :: !xs;
          skip_ws ()
        done;
        if !pos >= len then failwith "cert: unterminated list";
        incr pos;
        List (List.rev !xs)
    | '"' ->
        incr pos;
        let buf = Buffer.create 16 in
        while !pos < len && s.[!pos] <> '"' do
          if s.[!pos] = '\\' && !pos + 1 < len then begin
            (match s.[!pos + 1] with
             | 'n' -> Buffer.add_char buf '\n'
             | 't' -> Buffer.add_char buf '\t'
             | '\\' -> Buffer.add_char buf '\\'
             | '"' -> Buffer.add_char buf '"'
             | c -> Buffer.add_char buf c);
            pos := !pos + 2
          end else begin
            Buffer.add_char buf s.[!pos];
            incr pos
          end
        done;
        if !pos >= len then failwith "cert: unterminated string";
        incr pos;
        Atom (Buffer.contents buf)
    | _ ->
        let start = !pos in
        while !pos < len &&
              (let c = s.[!pos] in
               c <> '(' && c <> ')' && c <> '"' &&
               c <> ' ' && c <> '\t' && c <> '\n' && c <> '\r' && c <> ';') do
          incr pos
        done;
        Atom (String.sub s start (!pos - start))
  in
  let r = parse_one () in
  skip_ws ();
  r

(* ---------- Term / type micro-parser (kernel-side) ------------------- *)

(* The kernel accepts a *strictly* simple textual term syntax for use
   inside certificates:

     <type> := name | name(<type>, ...) | <type> -> <type> | 'name
     <term> := name                       (constant or variable, ambiguous)
             | name : <type>              (annotated variable)
             | (<term> <term>)
             | (\name : <type>. <term>)
             | (<term> = <term>)
             | (<term> /\ <term>)
             | (<term> \/ <term>)
             | (<term> ==> <term>)
             | (~ <term>)
             | (! name : <type>. <term>)
             | (? name : <type>. <term>)

   The textual form is unambiguous given enough parentheses; the
   prover or theory package is responsible for emitting fully
   parenthesised forms.  This keeps the kernel-side parser tiny. *)

let rec parse_type_pos s pos =
  let len = String.length s in
  let skip () =
    while !pos < len && (s.[!pos] = ' ' || s.[!pos] = '\t') do incr pos done
  in
  skip ();
  let lhs =
    if !pos < len && s.[!pos] = '\'' then begin
      incr pos;
      let st = !pos in
      while !pos < len &&
            (let c = s.[!pos] in
             c <> ' ' && c <> ')' && c <> ',' && c <> '-' && c <> '\t'
             && c <> '\n' && c <> '.' && c <> ':')
      do incr pos done;
      Type.Tyvar (String.sub s st (!pos - st))
    end else if !pos < len && s.[!pos] = '(' then begin
      incr pos;
      let t = parse_type_pos s pos in
      skip ();
      if !pos < len && s.[!pos] = ')' then begin incr pos; t end
      else failwith ("parse_type: missing ')' at " ^ string_of_int !pos)
    end else begin
      let st = !pos in
      while !pos < len &&
            (let c = s.[!pos] in
             c <> ' ' && c <> ')' && c <> ',' && c <> '(' && c <> '-'
             && c <> '\t' && c <> '\n' && c <> '.' && c <> ':')
      do incr pos done;
      let name = String.sub s st (!pos - st) in
      if name = "" then failwith ("parse_type: empty at " ^ string_of_int !pos);
      skip ();
      if !pos < len && s.[!pos] = '(' then begin
        incr pos;
        let args = ref [] in
        skip ();
        if !pos < len && s.[!pos] <> ')' then begin
          args := [parse_type_pos s pos];
          skip ();
          while !pos < len && s.[!pos] = ',' do
            incr pos;
            args := parse_type_pos s pos :: !args;
            skip ()
          done
        end;
        if !pos < len && s.[!pos] = ')' then incr pos
        else failwith "parse_type: missing ')'";
        Type.Tyapp (name, List.rev !args)
      end else
        Type.Tyapp (name, [])
    end
  in
  skip ();
  if !pos + 1 < len && s.[!pos] = '-' && s.[!pos + 1] = '>' then begin
    pos := !pos + 2;
    let r = parse_type_pos s pos in
    Type.fun_ty lhs r
  end else lhs

let parse_type s =
  let pos = ref 0 in
  let t = parse_type_pos s pos in
  t

(* Type registry must include the constants used in the cert.  For the
   MVP we register [nat] up front since the number-theory theory uses
   it; theory packages can add more via [Type.register_tyconstr]. *)
let () = Type.register_tyconstr "nat" 0

(* The term parser is intentionally minimal and fully-parenthesised.
   It threads an environment of bound names so that unannotated
   occurrences of a bound variable inherit its declared type. *)
let parse_term s =
  let pos = ref 0 in
  let len = String.length s in
  let skip () =
    while !pos < len &&
          (s.[!pos] = ' ' || s.[!pos] = '\t' || s.[!pos] = '\n') do
      incr pos done
  in
  let peek () = if !pos < len then Some s.[!pos] else None in
  let starts_with str =
    let n = String.length str in
    !pos + n <= len && String.sub s !pos n = str
  in
  let read_ident () =
    let st = !pos in
    while !pos < len &&
          (let c = s.[!pos] in
           c <> ' ' && c <> '\t' && c <> '\n' && c <> '(' && c <> ')' &&
           c <> ':' && c <> '.' && c <> ',') do
      incr pos
    done;
    String.sub s st (!pos - st)
  in
  let bool_bool_bool () =
    Type.fun_ty Type.bool_ty (Type.fun_ty Type.bool_ty Type.bool_ty)
  in
  let expect c =
    if !pos >= len || s.[!pos] <> c then
      failwith (Printf.sprintf "parse_term: expected %c at %d, found %s"
                  c !pos (String.sub s !pos (min 20 (len - !pos))));
    incr pos
  in
  let rec parse env =
    skip ();
    match peek () with
    | None -> failwith "parse_term: unexpected EOF"
    | Some '(' ->
        incr pos; skip ();
        (match peek () with
         | Some '!' ->
             incr pos; skip ();
             let name = read_ident () in
             skip ();
             expect ':';
             skip ();
             let ty = parse_type_pos s pos in
             skip (); expect '.';
             let body = parse ((name, ty) :: env) in
             skip (); expect ')';
             Rules.mk_forall (name, ty) body
         | Some '?' ->
             incr pos; skip ();
             let name = read_ident () in
             skip ();
             expect ':';
             skip ();
             let ty = parse_type_pos s pos in
             skip (); expect '.';
             let body = parse ((name, ty) :: env) in
             skip (); expect ')';
             Rules.mk_exists (name, ty) body
         | Some '\\' ->
             incr pos; skip ();
             let name = read_ident () in
             skip ();
             expect ':';
             skip ();
             let ty = parse_type_pos s pos in
             skip (); expect '.';
             let body = parse ((name, ty) :: env) in
             skip (); expect ')';
             Term.mk_abs (Term.Var (name, ty)) body
         | Some '~' when !pos + 1 < len && s.[!pos + 1] = ' ' ->
             incr pos; skip ();
             let body = parse env in
             skip (); expect ')';
             let not_const = Term.Const ("~", Type.fun_ty Type.bool_ty Type.bool_ty) in
             Term.mk_comb not_const body
         | _ ->
             let t1 = parse env in
             skip ();
             if starts_with "==>" then begin
               pos := !pos + 3; skip ();
               let t2 = parse env in
               skip (); expect ')';
               Rules.mk_imp t1 t2
             end else if starts_with "/\\" then begin
               pos := !pos + 2; skip ();
               let t2 = parse env in
               skip (); expect ')';
               Rules.mk_conj t1 t2
             end else if starts_with "\\/" then begin
               pos := !pos + 2; skip ();
               let t2 = parse env in
               skip (); expect ')';
               let or_const = Term.Const ("\\/", bool_bool_bool ()) in
               Term.mk_comb (Term.mk_comb or_const t1) t2
             end else if starts_with "=" then begin
               pos := !pos + 1; skip ();
               let t2 = parse env in
               skip (); expect ')';
               Term.mk_eq t1 t2
             end else begin
               let rec gather acc =
                 skip ();
                 if !pos < len && s.[!pos] = ')' then begin
                   incr pos; acc
                 end else
                   let nxt = parse env in
                   gather (Term.mk_comb acc nxt)
               in
               gather t1
             end)
    | Some _ ->
        let name = read_ident () in
        skip ();
        if !pos < len && s.[!pos] = ':' then begin
          incr pos; skip ();
          let ty = parse_type_pos s pos in
          Term.Var (name, ty)
        end else
          (* Resolution order: bound names, then registered constants,
             then default to bool variable. *)
          (match List.assoc_opt name env with
           | Some ty -> Term.Var (name, ty)
           | None ->
               (match Hashtbl.find_opt Term.constants name with
                | Some ty -> Term.Const (name, ty)
                | None -> Term.Var (name, Type.bool_ty)))
  in
  parse []

(* ---------- Sexp → cert AST ------------------------------------------ *)

let atom = function Atom s -> s | _ -> failwith "expected atom"
let list_ = function List xs -> xs | _ -> failwith "expected list"

let parse_witness = function
  | List [] -> W_none
  | List [Atom "term"; Atom s] -> W_term (parse_term s)
  | List [Atom "type"; Atom s] -> W_type (parse_type s)
  | List [Atom "var"; Atom n; Atom ty] -> W_var (n, parse_type ty)
  | List [Atom "axiom"; Atom n] -> W_axiom n
  | List [Atom "bound_and_witness";
          List [Atom "bound"; Atom n; Atom ty];
          List [Atom "witness"; Atom w]] ->
      W_bound_and_witness ((n, parse_type ty), parse_term w)
  | List (Atom "inst" :: pairs) ->
      W_inst (List.map (function
        | List [Atom "subst"; Atom v_term; Atom rhs] ->
            (parse_term v_term, parse_term rhs)
        | _ -> failwith "bad inst entry") pairs)
  | List (Atom "insttype" :: pairs) ->
      W_inst_type (List.map (function
        | List [Atom "subst"; Atom a; Atom ty] ->
            (a, parse_type ty)
        | _ -> failwith "bad insttype entry") pairs)
  | _ -> failwith "cert: bad witness"

let parse_step = function
  | List items ->
      let h = List.hd items in
      if atom h <> "step" then failwith "expected (step ...)";
      let rest = List.tl items in
      let id = int_of_string (atom (List.hd rest)) in
      let rest = List.tl rest in
      let find tag =
        try Some (List.find (function
          | List (Atom t :: _) -> t = tag
          | _ -> false) rest)
        with Not_found -> None
      in
      let rule =
        match find "rule" with
        | Some (List [_; Atom n]) -> n
        | _ -> failwith "step missing rule"
      in
      let witness =
        match find "witness" with
        | Some (List [_; w]) -> parse_witness w
        | Some (List (_ :: ws)) -> parse_witness (List ws)
        | _ -> W_none
      in
      let premises =
        match find "premises" with
        | Some (List (_ :: xs)) ->
            List.map (fun x -> int_of_string (atom x)) xs
        | _ -> []
      in
      let declared_concl =
        match find "concl" with
        | Some (List [_; Atom s]) -> Some (parse_term s)
        | _ -> None
      in
      { id; rule; witness; premises; declared_concl }
  | _ -> failwith "expected list"

let of_sexp = function
  | List (Atom "cert" :: rest) ->
      let steps_sexp, concl_sexp =
        let rev = List.rev rest in
        match rev with
        | (List [Atom "concl"; Atom c]) :: ss ->
            (List.rev ss, parse_term c)
        | _ -> failwith "cert: missing final (concl ...)"
      in
      let steps = List.map parse_step steps_sexp in
      { steps; concl = concl_sexp }
  | _ -> failwith "cert: expected (cert ...)"

let parse s = of_sexp (read_sexp s)

let parse_file path =
  let ic = open_in path in
  let n = in_channel_length ic in
  let buf = Bytes.create n in
  really_input ic buf 0 n;
  close_in ic;
  parse (Bytes.to_string buf)

(* ---------- Phi-file (.kf) parser ----------------------------------- *)

(* A .kf file is a sequence of S-expressions of two forms:
     (axiom <name> "<term>")        — declare a theory-package axiom
     (goal "<term>")                — the formula to verify
   Returns the list of declared axioms and the goal term. *)

let read_top_sexps s =
  let pos = ref 0 in
  let len = String.length s in
  let skip () =
    while !pos < len && (let c = s.[!pos] in
                         c = ' ' || c = '\t' || c = '\n' || c = '\r'
                         || c = ';') do
      if s.[!pos] = ';' then
        while !pos < len && s.[!pos] <> '\n' do incr pos done
      else incr pos
    done
  in
  let out = ref [] in
  skip ();
  while !pos < len do
    let start = !pos in
    let depth = ref 0 in
    let in_str = ref false in
    let continue = ref true in
    while !continue && !pos < len do
      let c = s.[!pos] in
      if !in_str then begin
        if c = '\\' && !pos + 1 < len then pos := !pos + 2
        else if c = '"' then begin in_str := false; incr pos end
        else incr pos
      end else if c = '"' then begin in_str := true; incr pos end
      else if c = '(' then begin incr depth; incr pos end
      else if c = ')' then begin
        decr depth; incr pos;
        if !depth = 0 then continue := false
      end else incr pos
    done;
    if !pos > start then
      out := read_sexp (String.sub s start (!pos - start)) :: !out;
    skip ()
  done;
  List.rev !out

let parse_phi_file path =
  let ic = open_in path in
  let n = in_channel_length ic in
  let buf = Bytes.create n in
  really_input ic buf 0 n;
  close_in ic;
  let sexps = read_top_sexps (Bytes.to_string buf) in
  let goal = ref None in
  let axioms = ref [] in
  (* Two passes: register types/constants first, then parse axioms and
     goal which may reference them. *)
  List.iter (fun sx ->
    match sx with
    | List [Atom "type"; Atom name] ->
        Type.register_tyconstr name 0
    | List [Atom "type"; Atom name; Atom arity] ->
        Type.register_tyconstr name (int_of_string arity)
    | List [Atom "const"; Atom name; Atom ty_str] ->
        Term.register_const name (parse_type ty_str)
    | _ -> ()
  ) sexps;
  List.iter (fun sx ->
    match sx with
    | List [Atom "axiom"; Atom name; Atom term_str] ->
        axioms := (name, parse_term term_str) :: !axioms
    | List [Atom "goal"; Atom term_str] ->
        goal := Some (parse_term term_str)
    | List [Atom "type"; _] | List [Atom "type"; _; _]
    | List [Atom "const"; _; _] -> ()
    | _ ->
        Printf.eprintf "warning: ignored .kf form\n%!"
  ) sexps;
  (List.rev !axioms, !goal)

(* ---------- Apply a step's rule -------------------------------------- *)

let apply_step (table : (int, Thm.t) Hashtbl.t) (step : step) : Thm.t =
  let prem i =
    match Hashtbl.find_opt table i with
    | Some t -> t
    | None -> failwith (Printf.sprintf "cert: step %d cites unknown premise %d"
                          step.id i)
  in
  match step.rule, step.witness, step.premises with
  | "REFL", W_term t, [] -> Rules.refl t
  | "TRANS", W_none, [a; b] -> Rules.trans (prem a) (prem b)
  | "MK_COMB", W_none, [a; b] -> Rules.mk_comb (prem a) (prem b)
  | "ABS", W_var (n, ty), [a] -> Rules.abs (Term.Var (n, ty)) (prem a)
  | "BETA", W_term t, [] -> Rules.beta t
  | "ASSUME", W_term t, [] -> Rules.assume t
  | "EQ_MP", W_none, [a; b] -> Rules.eq_mp (prem a) (prem b)
  | "DEDUCT_ANTISYM_RULE", W_none, [a; b] ->
      Rules.deduct_antisym (prem a) (prem b)
  | "INST", W_inst theta, [a] -> Rules.inst theta (prem a)
  | "INST_TYPE", W_inst_type theta, [a] -> Rules.inst_type theta (prem a)
  (* connective primitives *)
  | "GEN", W_var (n, ty), [a] -> Rules.gen (Term.Var (n, ty)) (prem a)
  | "SPEC", W_term t, [a] -> Rules.spec t (prem a)
  | "EXISTS", W_bound_and_witness ((n, ty), w), [a] ->
      Rules.exists_intro ~bound:(n, ty) ~witness:w (prem a)
  | "CHOOSE", W_var (n, ty), [a; b] ->
      Rules.choose ~witness:(n, ty) (prem a) (prem b)
  | "CONJ", W_none, [a; b] -> Rules.conj (prem a) (prem b)
  | "CONJUNCT1", W_none, [a] -> Rules.conjunct1 (prem a)
  | "CONJUNCT2", W_none, [a] -> Rules.conjunct2 (prem a)
  | "MP", W_none, [a; b] -> Rules.mp (prem a) (prem b)
  | "DISCH", W_term p, [a] -> Rules.disch p (prem a)
  (* axioms *)
  | "AXIOM", W_axiom name, [] -> Axioms.axiom_thm name
  | "ETA_AX", W_none, [] -> Axioms.eta_ax ()
  | "SELECT_AX", W_none, [] -> Axioms.select_ax ()
  | "EM_AX", W_none, [] -> Axioms.em_ax ()
  | _ ->
      failwith (Printf.sprintf "cert: step %d: bad rule/witness/premises: %s"
                  step.id step.rule)

(* ---------- S-exp printer (for tests / debugging) -------------------- *)

let pp_type = Type.to_string

let rec pp_term t = match t with
  | Term.Var (n, ty) -> n ^ ":" ^ pp_type ty
  | Term.Const (c, _) -> c
  | Term.Comb (Term.Comb (Term.Const ("=", _), a), b) ->
      "(" ^ pp_term a ^ " = " ^ pp_term b ^ ")"
  | Term.Comb (Term.Comb (Term.Const ("/\\", _), a), b) ->
      "(" ^ pp_term a ^ " /\\ " ^ pp_term b ^ ")"
  | Term.Comb (Term.Comb (Term.Const ("\\/", _), a), b) ->
      "(" ^ pp_term a ^ " \\/ " ^ pp_term b ^ ")"
  | Term.Comb (Term.Comb (Term.Const ("==>", _), a), b) ->
      "(" ^ pp_term a ^ " ==> " ^ pp_term b ^ ")"
  | Term.Comb (Term.Const ("!", _), Term.Abs (Term.Var (n, ty), b)) ->
      "(! " ^ n ^ " : " ^ pp_type ty ^ ". " ^ pp_term b ^ ")"
  | Term.Comb (Term.Const ("?", _), Term.Abs (Term.Var (n, ty), b)) ->
      "(? " ^ n ^ " : " ^ pp_type ty ^ ". " ^ pp_term b ^ ")"
  | Term.Comb (Term.Const ("~", _), b) -> "(~ " ^ pp_term b ^ ")"
  | Term.Comb (f, x) -> "(" ^ pp_term f ^ " " ^ pp_term x ^ ")"
  | Term.Abs (Term.Var (n, ty), b) ->
      "(\\ " ^ n ^ " : " ^ pp_type ty ^ ". " ^ pp_term b ^ ")"
  | Term.Abs _ -> "<bad-abs>"

(* tokenizer/lexicon.ml — token-ID assignments for the HOL certificate
   vocabulary.  Lexeme-level granularity, fixed size, fits uint8.

   Layout (see vocab_size below):
     0..3       special (BOS, EOS, PAD, UNK)
     4..15      structural punctuation
     16..23     term operators
     24..47     keywords (.thy / .kf / .cert / witness tags)
     48..71     rule names
     72..79     built-in type constructors
     80..87     type-variable pool
     88..103    type-constructor pool (theory-declared)
     104..167   name pool (constants + axiom names, theory-declared)
     168..199   variable pool (canonicalised in encode order)
     200..231   integer literal pool

   Constants and axiom names share one pool because they behave the
   same way at the token level (both are theory-scoped identifiers
   that the encoder allocates on first sight). *)

(* --- Special ---------------------------------------------------------- *)
let bos = 0
let eos = 1
let pad = 2
let unk = 3

(* --- Structural punctuation ------------------------------------------- *)
let lparen = 4
let rparen = 5
let colon  = 6
let dot    = 7
let comma  = 8
let quote  = 9
let arrow  = 10
(* 11..15 reserved *)

(* --- Term operators --------------------------------------------------- *)
let op_eq      = 16
let op_imp     = 17
let op_conj    = 18
let op_disj    = 19
let op_not     = 20
let op_forall  = 21
let op_exists  = 22
let op_lambda  = 23

(* --- Keywords (24..47, 24 slots, 19 used) ---------------------------- *)
let kw_type      = 24    (* (type ...) declaration AND (type "...") witness *)
let kw_const     = 25
let kw_axiom     = 26    (* (axiom ...) declaration AND (axiom "...") witness *)
let kw_theorem   = 27
let kw_goal      = 28
let kw_cert      = 29
let kw_step      = 30
let kw_rule      = 31
let kw_witness   = 32
let kw_premises  = 33
let kw_concl     = 34
let kw_term      = 35    (* (term "...") witness *)
let kw_var       = 36    (* (var "n" "ty") witness *)
let kw_inst      = 37
let kw_insttype  = 38
let kw_b_and_w   = 39    (* (bound_and_witness ...) *)
let kw_subst     = 40
let kw_bound     = 41
(* 42..47 reserved *)

(* --- Rule names (48..71, 24 slots, 23 used) -------------------------- *)
let rule_refl                = 48
let rule_trans               = 49
let rule_mk_comb             = 50
let rule_abs                 = 51
let rule_beta                = 52
let rule_assume              = 53
let rule_eq_mp               = 54
let rule_deduct_antisym_rule = 55
let rule_inst                = 56
let rule_inst_type           = 57
let rule_axiom               = 58
let rule_gen                 = 59
let rule_spec                = 60
let rule_exists              = 61
let rule_choose              = 62
let rule_conj                = 63
let rule_conjunct1           = 64
let rule_conjunct2           = 65
let rule_mp                  = 66
let rule_disch               = 67
let rule_eta_ax              = 68
let rule_select_ax           = 69
let rule_em_ax               = 70
(* 71 reserved *)

let rule_name_table = [|
  "REFL"; "TRANS"; "MK_COMB"; "ABS"; "BETA"; "ASSUME"; "EQ_MP";
  "DEDUCT_ANTISYM_RULE"; "INST"; "INST_TYPE"; "AXIOM"; "GEN"; "SPEC";
  "EXISTS"; "CHOOSE"; "CONJ"; "CONJUNCT1"; "CONJUNCT2"; "MP"; "DISCH";
  "ETA_AX"; "SELECT_AX"; "EM_AX"
|]

let rule_first = rule_refl
let rule_count = 23
let rule_last  = rule_first + rule_count - 1

let rule_of_name = function
  | "REFL"                -> Some rule_refl
  | "TRANS"               -> Some rule_trans
  | "MK_COMB"             -> Some rule_mk_comb
  | "ABS"                 -> Some rule_abs
  | "BETA"                -> Some rule_beta
  | "ASSUME"              -> Some rule_assume
  | "EQ_MP"               -> Some rule_eq_mp
  | "DEDUCT_ANTISYM_RULE" -> Some rule_deduct_antisym_rule
  | "INST"                -> Some rule_inst
  | "INST_TYPE"           -> Some rule_inst_type
  | "AXIOM"               -> Some rule_axiom
  | "GEN"                 -> Some rule_gen
  | "SPEC"                -> Some rule_spec
  | "EXISTS"              -> Some rule_exists
  | "CHOOSE"              -> Some rule_choose
  | "CONJ"                -> Some rule_conj
  | "CONJUNCT1"           -> Some rule_conjunct1
  | "CONJUNCT2"           -> Some rule_conjunct2
  | "MP"                  -> Some rule_mp
  | "DISCH"               -> Some rule_disch
  | "ETA_AX"              -> Some rule_eta_ax
  | "SELECT_AX"           -> Some rule_select_ax
  | "EM_AX"               -> Some rule_em_ax
  | _                     -> None

let name_of_rule id =
  if id < rule_first || id > rule_last then None
  else Some rule_name_table.(id - rule_first)

(* --- Built-in type constructors (72..79) ----------------------------- *)
let ty_bool = 72
let ty_ind  = 73
let ty_fun  = 74
let ty_nat  = 75
(* 76..79 reserved for additional builtins *)

let builtin_type_table = [|
  "bool"; "ind"; "fun"; "nat"
|]

let builtin_of_name = function
  | "bool" -> Some ty_bool
  | "ind"  -> Some ty_ind
  | "fun"  -> Some ty_fun
  | "nat"  -> Some ty_nat
  | _      -> None

let name_of_builtin id =
  if id < ty_bool || id >= ty_bool + Array.length builtin_type_table then None
  else Some builtin_type_table.(id - ty_bool)

(* --- Type-variable pool (80..87) ------------------------------------- *)
let tyvar_first = 80
let tyvar_count = 8
let tyvar_last  = tyvar_first + tyvar_count - 1
let tyvar_tok k = tyvar_first + k

(* --- Type-constructor pool (88..103) --------------------------------- *)
let tycon_first = 88
let tycon_count = 16
let tycon_last  = tycon_first + tycon_count - 1
let tycon_tok k = tycon_first + k

(* --- Name pool: constants + axiom names (104..167) ------------------- *)
let name_first = 104
let name_count = 64
let name_last  = name_first + name_count - 1
let name_tok k = name_first + k

(* --- Variable pool (168..199) ---------------------------------------- *)
let var_first = 168
let var_count = 32
let var_last  = var_first + var_count - 1
let var_tok k = var_first + k

(* --- Integer literal pool (200..231) --------------------------------- *)
let int_first = 200
let int_count = 32
let int_last  = int_first + int_count - 1
let int_tok k = int_first + k

(* --- Vocabulary size ------------------------------------------------- *)
let vocab_size = 232

(* --- Token classification helpers ------------------------------------ *)
let is_special id  = id <= unk
let is_rule id     = id >= rule_first  && id <= rule_last
let is_builtin_ty id = id >= ty_bool   && id < ty_bool + Array.length builtin_type_table
let is_tyvar id    = id >= tyvar_first && id <= tyvar_last
let is_tycon id    = id >= tycon_first && id <= tycon_last
let is_name id     = id >= name_first  && id <= name_last
let is_var id      = id >= var_first   && id <= var_last
let is_int id      = id >= int_first   && id <= int_last

let int_of_int_tok id =
  if is_int id then Some (id - int_first) else None

let var_index id =
  if is_var id then Some (id - var_first) else None

let name_index id =
  if is_name id then Some (id - name_first) else None

let tycon_index id =
  if is_tycon id then Some (id - tycon_first) else None

let tyvar_index id =
  if is_tyvar id then Some (id - tyvar_first) else None

(* --- Pretty-print for debugging -------------------------------------- *)
let to_string id =
  if id = bos then "BOS"
  else if id = eos then "EOS"
  else if id = pad then "PAD"
  else if id = unk then "UNK"
  else if id = lparen then "("
  else if id = rparen then ")"
  else if id = colon  then ":"
  else if id = dot    then "."
  else if id = comma  then ","
  else if id = quote  then "\""
  else if id = arrow  then "->"
  else if id = op_eq then "="
  else if id = op_imp then "==>"
  else if id = op_conj then "/\\"
  else if id = op_disj then "\\/"
  else if id = op_not then "~"
  else if id = op_forall then "!"
  else if id = op_exists then "?"
  else if id = op_lambda then "\\"
  else if id = kw_type then "KW_type"
  else if id = kw_const then "KW_const"
  else if id = kw_axiom then "KW_axiom"
  else if id = kw_theorem then "KW_theorem"
  else if id = kw_goal then "KW_goal"
  else if id = kw_cert then "KW_cert"
  else if id = kw_step then "KW_step"
  else if id = kw_rule then "KW_rule"
  else if id = kw_witness then "KW_witness"
  else if id = kw_premises then "KW_premises"
  else if id = kw_concl then "KW_concl"
  else if id = kw_term then "KW_term"
  else if id = kw_var then "KW_var"
  else if id = kw_inst then "KW_inst"
  else if id = kw_insttype then "KW_insttype"
  else if id = kw_b_and_w then "KW_b_and_w"
  else if id = kw_subst then "KW_subst"
  else if id = kw_bound then "KW_bound"
  else match name_of_rule id with
    | Some n -> "RULE_" ^ n
    | None ->
      match name_of_builtin id with
      | Some n -> "TY_" ^ n
      | None ->
        if is_tyvar id then Printf.sprintf "TYVAR_a%d" (id - tyvar_first)
        else if is_tycon id then Printf.sprintf "TYCON_t%d" (id - tycon_first)
        else if is_name id then Printf.sprintf "NAME_n%d" (id - name_first)
        else if is_var id then Printf.sprintf "VAR_v%d" (id - var_first)
        else if is_int id then Printf.sprintf "INT_%d" (id - int_first)
        else Printf.sprintf "?<%d>" id

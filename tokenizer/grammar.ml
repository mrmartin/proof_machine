(* tokenizer/grammar.ml — pushdown automaton for the encoder's token
   language.  The state is a stack of "expected symbols"; [step] consumes
   one token and rewrites the top of the stack; [valid_next_mask] lists
   every token that [step] would accept.

   The grammar mirrors the encoder in [encode.ml] — for any Cert.t,
   the token stream produced by [Encode.cert] is accepted from
   [initial] and reaches [is_accepting] after the final RPAREN.

   The grammar is **sound** (never excludes a token the encoder emits)
   but only modestly tight: for open-vocab classes (variable / name /
   integer pools) we allow any in-range slot, not just the slots
   already allocated. *)

module L = Lexicon

type sym =
  | Tok of int                  (* exact terminal *)
  | TokIn of int list           (* any one of these terminals *)
  | TRule                       (* any rule-name token *)
  | TInt                        (* any int-literal token *)
  | TVar                        (* any variable-pool token *)
  | TName                       (* any name-pool token *)
  | TTyHead                     (* a leaf type head: builtin, tycon, or tyvar (for non-paren type position) *)
  | TTyvar                      (* a type-variable token *)
  | NTerm
  | NAfterLPTerm
  | NAfterFirstTerm
  | NType
  | NAfterLPType
  | NAfterFirstType
  | NCertBody                   (* zero+ ( KW_step ... ) then ( KW_concl ... ) *)
  | NStepOrConclAfterLP         (* after LPAREN inside cert body — KW_step vs KW_concl *)
  | NWitnessBody                (* after (KW_witness — RPAREN or LPAREN tag *)
  | NWitnessTagAfterLP          (* after LPAREN in witness body — pick tag *)
  | NInstPairs                  (* zero+ (subst "t" "t") then RPAREN *)
  | NInstTypePairs
  | NPremiseList
  | NStepBody                   (* after (KW_step id — fields then RPAREN *)
  | NStepFieldAfterLP           (* after LPAREN inside step body — pick KW_rule|KW_witness|KW_premises *)
  | NDone

type state = sym list

let initial : state = [
  Tok L.lparen;
  Tok L.kw_cert;
  NCertBody;
  Tok L.rparen;
  NDone;
]

let is_accepting (s : state) =
  match s with
  | [NDone] | [] -> true
  | _ -> false

(* --- Helpers --------------------------------------------------------- *)

let leaf_tycon_or_builtin =
  let xs = ref [] in
  for i = L.tycon_first to L.tycon_last do xs := i :: !xs done;
  xs := L.ty_bool :: L.ty_ind :: L.ty_fun :: L.ty_nat :: !xs;
  !xs

let leaf_type_heads_paren =
  (* what can sit *after* the RPAREN closing a type's arg list *)
  leaf_tycon_or_builtin

let any_name_pool =
  let xs = ref [] in
  for i = L.name_first to L.name_last do xs := i :: !xs done;
  !xs

let any_var_pool =
  let xs = ref [] in
  for i = L.var_first to L.var_last do xs := i :: !xs done;
  !xs

let any_int_pool =
  let xs = ref [] in
  for i = L.int_first to L.int_last do xs := i :: !xs done;
  !xs

let any_rule =
  let xs = ref [] in
  for i = L.rule_first to L.rule_last do xs := i :: !xs done;
  !xs

let any_tyvar_pool =
  let xs = ref [] in
  for i = L.tyvar_first to L.tyvar_last do xs := i :: !xs done;
  !xs

let witness_tags =
  [L.kw_term; L.kw_type; L.kw_var; L.kw_axiom;
   L.kw_inst; L.kw_insttype; L.kw_b_and_w]

let step_field_kws = [L.kw_rule; L.kw_witness; L.kw_premises]

let binder_ops = [L.op_forall; L.op_exists; L.op_lambda]

let binary_ops = [L.op_eq; L.op_conj; L.op_disj; L.op_imp]

(* --- Step ------------------------------------------------------------ *)

(* Consume one token.  Returns the new stack, or None on rejection.
   Internally we may expand non-terminals until the top is a terminal
   that the current token can match against; the expansion choices are
   driven by lookahead at the supplied token. *)

let rec step (stack : state) (tok : int) : state option =
  match stack with
  | [] -> None
  | Tok t :: rest ->
    if tok = t then Some rest else None
  | TokIn ts :: rest ->
    if List.mem tok ts then Some rest else None
  | TRule :: rest ->
    if L.is_rule tok then Some rest else None
  | TInt :: rest ->
    if L.is_int tok then Some rest else None
  | TVar :: rest ->
    if L.is_var tok then Some rest else None
  | TName :: rest ->
    if L.is_name tok then Some rest else None
  | TTyHead :: rest ->
    if L.is_builtin_ty tok || L.is_tycon tok || L.is_tyvar tok then Some rest
    else None
  | TTyvar :: rest ->
    if L.is_tyvar tok then Some rest else None

  | NTerm :: rest ->
    if tok = L.lparen then step (Tok L.lparen :: NAfterLPTerm :: rest) tok
    else if L.is_var tok then
      step (TVar :: Tok L.colon :: NType :: rest) tok
    else if L.is_name tok then
      step (TName :: rest) tok
    else if tok = L.op_eq then
      step (Tok L.op_eq :: rest) tok
    else None

  | NAfterLPTerm :: rest ->
    if List.mem tok binder_ops then
      step (TokIn binder_ops
            :: TVar :: Tok L.colon :: NType :: Tok L.dot :: NTerm
            :: Tok L.rparen :: rest) tok
    else if tok = L.op_not then
      step (Tok L.op_not :: NTerm :: Tok L.rparen :: rest) tok
    else
      (* It's the first inner sub-term of either an application or a
         binary-op form.  Replace with NTerm :: NAfterFirstTerm. *)
      step (NTerm :: NAfterFirstTerm :: rest) tok

  | NAfterFirstTerm :: rest ->
    if List.mem tok binary_ops then
      step (TokIn binary_ops :: NTerm :: Tok L.rparen :: rest) tok
    else
      (* Application form: second sub-term, then RPAREN. *)
      step (NTerm :: Tok L.rparen :: rest) tok

  | NType :: rest ->
    if tok = L.lparen then
      step (Tok L.lparen :: NAfterLPType :: rest) tok
    else if L.is_tyvar tok || L.is_builtin_ty tok || L.is_tycon tok then
      step (TTyHead :: rest) tok
    else None

  | NAfterLPType :: rest ->
    (* Both fun-types and N-ary tyapps start with a type. *)
    step (NType :: NAfterFirstType :: rest) tok

  | NAfterFirstType :: rest ->
    if tok = L.arrow then
      step (Tok L.arrow :: NType :: Tok L.rparen :: rest) tok
    else if tok = L.comma then
      step (Tok L.comma :: NType :: NAfterFirstType :: rest) tok
    else if tok = L.rparen then
      (* End of arg list (single-arg or end of multi-arg).  After RPAREN
         we need a type head. *)
      step (Tok L.rparen
            :: TokIn (L.ty_bool :: L.ty_ind :: L.ty_fun :: L.ty_nat ::
                      (let xs = ref [] in
                       for i = L.tycon_first to L.tycon_last do xs := i :: !xs done;
                       !xs))
            :: rest) tok
    else None

  | NCertBody :: rest ->
    (* A series of zero or more (step ...) entries followed by one
       (concl ...) block.  Disambiguate via the keyword after LPAREN. *)
    if tok = L.lparen then
      step (Tok L.lparen :: NStepOrConclAfterLP :: rest) tok
    else None

  | NStepOrConclAfterLP :: rest ->
    if tok = L.kw_step then
      step (Tok L.kw_step :: TInt :: NStepBody :: Tok L.rparen
            :: NCertBody :: rest) tok
    else if tok = L.kw_concl then
      step (Tok L.kw_concl :: Tok L.quote :: NTerm :: Tok L.quote :: Tok L.rparen
            :: rest) tok
    else None

  | NStepBody :: rest ->
    if tok = L.lparen then
      step (Tok L.lparen :: NStepFieldAfterLP :: NStepBody :: rest) tok
    else if tok = L.rparen then
      step (rest) tok
    else None

  | NStepFieldAfterLP :: rest ->
    if tok = L.kw_rule then
      step (Tok L.kw_rule :: TRule :: Tok L.rparen :: rest) tok
    else if tok = L.kw_witness then
      step (Tok L.kw_witness :: NWitnessBody :: rest) tok
    else if tok = L.kw_premises then
      step (Tok L.kw_premises :: NPremiseList :: rest) tok
    else None

  | NWitnessBody :: rest ->
    if tok = L.rparen then
      step (Tok L.rparen :: rest) tok
    else if tok = L.lparen then
      step (Tok L.lparen :: NWitnessTagAfterLP :: Tok L.rparen :: rest) tok
    else None

  | NWitnessTagAfterLP :: rest ->
    if tok = L.kw_term then
      step (Tok L.kw_term :: Tok L.quote :: NTerm :: Tok L.quote
            :: Tok L.rparen :: rest) tok
    else if tok = L.kw_type then
      step (Tok L.kw_type :: Tok L.quote :: NType :: Tok L.quote
            :: Tok L.rparen :: rest) tok
    else if tok = L.kw_var then
      step (Tok L.kw_var
            :: Tok L.quote :: TVar :: Tok L.quote
            :: Tok L.quote :: NType :: Tok L.quote
            :: Tok L.rparen :: rest) tok
    else if tok = L.kw_axiom then
      step (Tok L.kw_axiom :: Tok L.quote :: TName :: Tok L.quote
            :: Tok L.rparen :: rest) tok
    else if tok = L.kw_inst then
      step (Tok L.kw_inst :: NInstPairs :: rest) tok
    else if tok = L.kw_insttype then
      step (Tok L.kw_insttype :: NInstTypePairs :: rest) tok
    else if tok = L.kw_b_and_w then
      step (Tok L.kw_b_and_w
            :: Tok L.lparen :: Tok L.kw_bound
              :: Tok L.quote :: TVar :: Tok L.quote
              :: Tok L.quote :: NType :: Tok L.quote
              :: Tok L.rparen
            :: Tok L.lparen :: Tok L.kw_witness
              :: Tok L.quote :: NTerm :: Tok L.quote
              :: Tok L.rparen
            :: Tok L.rparen :: rest) tok
    else None

  | NInstPairs :: rest ->
    if tok = L.rparen then
      Some rest
    else if tok = L.lparen then
      step (Tok L.lparen :: Tok L.kw_subst
            :: Tok L.quote :: NTerm :: Tok L.quote
            :: Tok L.quote :: NTerm :: Tok L.quote
            :: Tok L.rparen :: NInstPairs :: rest) tok
    else None

  | NInstTypePairs :: rest ->
    if tok = L.rparen then
      Some rest
    else if tok = L.lparen then
      step (Tok L.lparen :: Tok L.kw_subst
            :: Tok L.quote :: TVar :: Tok L.quote
            :: Tok L.quote :: NType :: Tok L.quote
            :: Tok L.rparen :: NInstTypePairs :: rest) tok
    else None

  | NPremiseList :: rest ->
    if tok = L.rparen then
      Some rest
    else if L.is_int tok then
      step (TInt :: NPremiseList :: rest) tok
    else None

  | NDone :: rest ->
    if tok = L.bos || tok = L.eos then Some rest else None

(* --- Mask ------------------------------------------------------------ *)

(* Returns the set of token IDs for which [step] would succeed.
   Sound (every accepted continuation is in the set) but not always
   tight (open-vocab pools admit any in-range slot). *)
let valid_next_set (stack : state) : int list =
  let rec collect stack =
    match stack with
    | [] -> []
    | Tok t :: _ -> [t]
    | TokIn ts :: _ -> ts
    | TRule :: _ -> any_rule
    | TInt :: _ -> any_int_pool
    | TVar :: _ -> any_var_pool
    | TName :: _ -> any_name_pool
    | TTyHead :: _ ->
      let xs = ref [] in
      for i = L.tycon_first to L.tycon_last do xs := i :: !xs done;
      L.ty_bool :: L.ty_ind :: L.ty_fun :: L.ty_nat ::
      (any_tyvar_pool @ !xs)
    | TTyvar :: _ -> any_tyvar_pool
    | NTerm :: _ ->
      L.lparen :: L.op_eq :: (any_var_pool @ any_name_pool)
    | NAfterLPTerm :: rest ->
      (* binder, negation, or first sub-term *)
      binder_ops @ (L.op_not :: collect (NTerm :: rest))
    | NAfterFirstTerm :: rest ->
      binary_ops @ collect (NTerm :: rest)
    | NType :: _ ->
      L.lparen :: L.ty_bool :: L.ty_ind :: L.ty_fun :: L.ty_nat ::
      (let xs = ref [] in
       for i = L.tycon_first to L.tycon_last do xs := i :: !xs done;
       any_tyvar_pool @ !xs)
    | NAfterLPType :: rest -> collect (NType :: rest)
    | NAfterFirstType :: _ -> [L.arrow; L.comma; L.rparen]
    | NCertBody :: _ -> [L.lparen]
    | NStepOrConclAfterLP :: _ -> [L.kw_step; L.kw_concl]
    | NStepBody :: _ -> [L.lparen; L.rparen]
    | NStepFieldAfterLP :: _ -> step_field_kws
    | NWitnessBody :: _ -> [L.lparen; L.rparen]
    | NWitnessTagAfterLP :: _ -> witness_tags
    | NInstPairs :: _ -> [L.lparen; L.rparen]
    | NInstTypePairs :: _ -> [L.lparen; L.rparen]
    | NPremiseList :: _ -> L.rparen :: any_int_pool
    | NDone :: _ -> [L.bos; L.eos]
  in
  collect stack

let valid_next_mask (stack : state) : bool array =
  let m = Array.make L.vocab_size false in
  List.iter (fun t -> if t >= 0 && t < L.vocab_size then m.(t) <- true)
    (valid_next_set stack);
  m

(* bin/verify_tokens.ml — persistent kernel verifier worker.

   Speaks a simple line-oriented protocol on stdin/stdout so a single
   process can serve many verification requests without paying fork+exec
   per call.

   Protocol — one request per stdin line:

     <cert_len> <cert_tok_0> ... <cert_tok_{n-1}> <goal_len> <goal_tok_0> ... <goal_tok_{m-1}>

   Where each token is a non-negative decimal integer in
   [0, Tokenizer.Lexicon.vocab_size).  Whitespace-separated.

   One response line per request:

       100   — cert kernel-verifies AND its concl is alpha-equiv to goal
       0     — cert kernel-verifies but its concl ≠ goal (proves wrong thm)
      -1     — cert doesn't decode, kernel rejects it, or any other error

   The verifier registers `nat` as a 0-arity type constructor up front,
   matching the encoder's assumption.  No other theory state. *)

open Kernel
open Tokenizer

(* --- Setup ---------------------------------------------------------- *)

let () =
  if not (Type.well_formed (Type.Tyapp ("nat", []))) then
    Type.register_tyconstr "nat" 0

(* --- Header inference from a flat token list ------------------------ *)

let infer_header (toks : int array) : Encode.pool_header =
  let vars   = ref [] in
  let names  = ref [] in
  let tycons = ref [] in
  let tyvars = ref [] in
  let alloc_slot lst k prefix =
    let cur = List.length !lst in
    if cur <= k then
      for i = cur to k do
        lst := !lst @ [Printf.sprintf "%s%d" prefix i]
      done
  in
  Array.iter (fun t ->
    if Lexicon.is_var t then
      alloc_slot vars (t - Lexicon.var_first) "v"
    else if Lexicon.is_name t then
      alloc_slot names (t - Lexicon.name_first) "n"
    else if Lexicon.is_tycon t then
      alloc_slot tycons (t - Lexicon.tycon_first) "t"
    else if Lexicon.is_tyvar t then
      alloc_slot tyvars (t - Lexicon.tyvar_first) "a"
  ) toks;
  { Encode.tycons = Array.of_list !tycons;
    tyvars        = Array.of_list !tyvars;
    names         = Array.of_list !names;
    vars          = Array.of_list !vars }

(* --- Term decoding (the goal is a bare term, not a cert) ----------- *)

let decode_term_array (hdr : Encode.pool_header) (toks : int array) : Term.term =
  let c = { Decode.toks; pos = 0 } in
  Decode.decode_term hdr c

(* Note: we don't canonicalise free var names here.  Both cert and goal
   are decoded with the same `infer_header` scheme, so the first free var
   in each gets slot 0 → name "v0", the second slot 1 → "v1", etc.
   Direct [Term.alpha_eq] is then the right comparator. *)

(* --- One request --------------------------------------------------- *)

let handle_request (line : string) : int =
  try
    let parts =
      String.split_on_char ' ' line
      |> List.filter (fun s -> s <> "")
      |> List.map int_of_string
    in
    let arr = Array.of_list parts in
    if Array.length arr < 1 then -1
    else
      let cert_len = arr.(0) in
      if cert_len < 0 || cert_len + 1 + 1 > Array.length arr then -1
      else
        let cert_toks = Array.sub arr 1 cert_len in
        let goal_len_idx = 1 + cert_len in
        let goal_len = arr.(goal_len_idx) in
        if goal_len < 0
           || goal_len_idx + 1 + goal_len > Array.length arr
        then -1
        else
          let goal_toks = Array.sub arr (goal_len_idx + 1) goal_len in
          let cert_hdr = infer_header cert_toks in
          let goal_hdr = infer_header goal_toks in
          match
            try
              let cert = Decode.cert cert_hdr cert_toks in
              let goal = decode_term_array goal_hdr goal_toks in
              Some (cert, goal)
            with _ -> None
          with
          | None -> -1
          | Some (cert, goal) ->
            (* Kernel verify the cert against ITS OWN declared concl. *)
            (match Verify.verify cert cert.concl with
             | Verify.Ok ->
               if Term.alpha_eq cert.concl goal then 100 else 0
             | Verify.Reject _ -> -1)
  with _ -> -1

(* --- Main loop ----------------------------------------------------- *)

let () =
  set_binary_mode_in stdin false;
  set_binary_mode_out stdout false;
  try
    while true do
      let line = input_line stdin in
      let r = handle_request line in
      print_string (string_of_int r);
      print_char '\n';
      (* Force flush after every response so the Python parent can read. *)
      flush stdout
    done
  with End_of_file -> ()

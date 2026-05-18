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

(* Override the inferred name slots in a pool header with caller-supplied
   names.  Slots beyond the supplied list keep their synthetic "nK"
   placeholders.  This is the hook that lets the verifier recover the
   original constant / axiom names that the tokeniser canonicalises away. *)
let override_names (hdr : Encode.pool_header) (names : string list) : Encode.pool_header =
  let names_arr = Array.of_list names in
  let new_names = Array.copy hdr.names in
  let n = min (Array.length names_arr) (Array.length new_names) in
  for i = 0 to n - 1 do new_names.(i) <- names_arr.(i) done;
  { hdr with names = new_names }

(* Parse either the legacy protocol (cert_len cert_toks goal_len goal_toks)
   or the extended one (H n_cn <cn0..> n_gn <gn0..> cert_len cert_toks ...).
   The extended form lets the client pass the *original* NAME-pool entries
   so that constants and axiom names round-trip through verification.

   A third mode (prefix verification) is prefixed with "P <k>" and skips
   the goal-tokens field — the response is a structured readback of the
   theorem table after [k] steps.  See [handle_prefix_request] below. *)
let handle_request (line : string) : int =
  try
    let parts =
      String.split_on_char ' ' line
      |> List.filter (fun s -> s <> "")
    in
    let arr = Array.of_list parts in
    let n = Array.length arr in
    let idx = ref 0 in
    let pop_str () =
      if !idx >= n then failwith "short" else
      let s = arr.(!idx) in incr idx; s
    in
    let pop_int () = int_of_string (pop_str ()) in
    let pop_names () =
      let k = pop_int () in
      let lst = ref [] in
      for _ = 1 to k do lst := pop_str () :: !lst done;
      List.rev !lst
    in
    let pop_toks () =
      let k = pop_int () in
      Array.init k (fun _ -> pop_int ())
    in
    let (cert_names, goal_names) =
      if n > 0 && arr.(0) = "H" then begin
        incr idx;  (* consume 'H' *)
        let cn = pop_names () in
        let gn = pop_names () in
        (cn, gn)
      end else ([], [])
    in
    let cert_toks = pop_toks () in
    let goal_toks = pop_toks () in
    let cert_hdr = override_names (infer_header cert_toks) cert_names in
    let goal_hdr = override_names (infer_header goal_toks) goal_names in
    match
      try
        let cert = Decode.cert cert_hdr cert_toks in
        let goal = decode_term_array goal_hdr goal_toks in
        Some (cert, goal)
      with _ -> None
    with
    | None -> -1
    | Some (cert, goal) ->
      (match Verify.verify cert cert.concl with
       | Verify.Ok ->
         if Term.alpha_eq cert.concl goal then 100 else 0
       | Verify.Reject _ -> -1)
  with _ -> -1

(* --- Prefix-mode response -------------------------------------------- *)

(* Serialise a list of theorems against a starting pool header, emitting:

     OK <k> <step_id_0> <n_hyps_0> <hyp_0_len_0> <hyp_0_toks_0...> ...
            <concl_len_0> <concl_toks_0...> <step_id_1> ...

   Slots not present in [start_hdr] are appended on first sight so each
   subsequent encoding sees the cumulative pool.  Token IDs round-trip
   against the original cert pool exactly when every name in derived
   theorems traces back to a witness in the cert (true for the rule set
   we support).

   Length is unbounded in principle but in practice bounded by the
   block-size limit the trainer enforces upstream; the response is one
   line. *)
let serialise_prefix_ok (start_hdr : Encode.pool_header)
                        (thms : (int * Thm.t) list) : string =
  let buf = Buffer.create 256 in
  Buffer.add_string buf "OK";
  Buffer.add_char buf ' ';
  Buffer.add_string buf (string_of_int (List.length thms));
  let emit_toks arr =
    Buffer.add_char buf ' ';
    Buffer.add_string buf (string_of_int (Array.length arr));
    Array.iter (fun t ->
      Buffer.add_char buf ' ';
      Buffer.add_string buf (string_of_int t)) arr
  in
  let hdr = ref start_hdr in
  List.iter (fun (step_id, thm) ->
    Buffer.add_char buf ' ';
    Buffer.add_string buf (string_of_int step_id);
    let hyps = Thm.hyps thm in
    Buffer.add_char buf ' ';
    Buffer.add_string buf (string_of_int (List.length hyps));
    List.iter (fun h ->
      let (toks, hdr') = Encode.term_with_header !hdr h in
      hdr := hdr';
      emit_toks toks) hyps;
    let (toks, hdr') = Encode.term_with_header !hdr (Thm.concl thm) in
    hdr := hdr';
    emit_toks toks) thms;
  Buffer.contents buf

let handle_prefix_request (line : string) : string =
  try
    let parts =
      String.split_on_char ' ' line
      |> List.filter (fun s -> s <> "")
    in
    let arr = Array.of_list parts in
    let n = Array.length arr in
    let idx = ref 0 in
    let pop_str () =
      if !idx >= n then failwith "short" else
      let s = arr.(!idx) in incr idx; s
    in
    let pop_int () = int_of_string (pop_str ()) in
    let pop_names () =
      let k = pop_int () in
      let lst = ref [] in
      for _ = 1 to k do lst := pop_str () :: !lst done;
      List.rev !lst
    in
    let pop_toks () =
      let k = pop_int () in
      Array.init k (fun _ -> pop_int ())
    in
    (* Consume the leading 'P' marker and the prefix length. *)
    if !idx >= n || arr.(!idx) <> "P" then failwith "not a prefix request";
    incr idx;
    let k = pop_int () in
    let cert_names =
      if !idx < n && arr.(!idx) = "H" then begin
        incr idx;
        let cn = pop_names () in
        let _ : string list = pop_names () in  (* goal-names slot unused *)
        cn
      end else []
    in
    let cert_toks = pop_toks () in
    let cert_hdr = override_names (infer_header cert_toks) cert_names in
    match
      try Some (Decode.cert cert_hdr cert_toks)
      with _ -> None
    with
    | None -> "ERR -1 decode_failed"
    | Some cert ->
      (match Verify.verify_prefix cert k with
       | Verify.Prefix_reject (step_i, _msg) ->
         Printf.sprintf "ERR %d" step_i
       | Verify.Prefix_ok thms -> serialise_prefix_ok cert_hdr thms)
  with _ -> "ERR -1 parse_failed"

(* --- Main loop ----------------------------------------------------- *)

(* Dispatch is based on the leading non-whitespace token.  Prefix-mode
   requests start with "P "; the legacy and H-prefixed protocols are
   unchanged and respond with a single integer. *)
let is_prefix_request (line : string) : bool =
  let len = String.length line in
  let rec skip_ws i =
    if i >= len then i
    else match line.[i] with
      | ' ' | '\t' -> skip_ws (i + 1)
      | _ -> i
  in
  let s = skip_ws 0 in
  s < len && line.[s] = 'P'
  && (s + 1 = len || line.[s + 1] = ' ' || line.[s + 1] = '\t')

let () =
  set_binary_mode_in stdin false;
  set_binary_mode_out stdout false;
  try
    while true do
      let line = input_line stdin in
      let response =
        if is_prefix_request line then handle_prefix_request line
        else string_of_int (handle_request line)
      in
      print_string response;
      print_char '\n';
      (* Force flush after every response so the Python parent can read. *)
      flush stdout
    done
  with End_of_file -> ()

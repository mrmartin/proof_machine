(* tokenizer/tests/test_tokenizer.ml — round-trip, grammar-accept,
   and mask-soundness tests for the tokenizer library. *)

open Kernel
open Tokenizer

(* Register the Euclid theory's constants up front so the decoder can
   reconstruct Const(name, ty) terms for the bundled Euclid corpus. *)
let () =
  let nat = Type.Tyapp ("nat", []) in
  if not (Type.well_formed nat) then Type.register_tyconstr "nat" 0;
  Term.register_const "one"       nat;
  Term.register_const "plus"      (Type.fun_ty nat (Type.fun_ty nat nat));
  Term.register_const "factorial" (Type.fun_ty nat nat);
  Term.register_const "prime"     (Type.fun_ty nat Type.bool_ty);
  Term.register_const "divides"   (Type.fun_ty nat (Type.fun_ty nat Type.bool_ty));
  Term.register_const "gt"        (Type.fun_ty nat (Type.fun_ty nat Type.bool_ty));
  Term.register_const "ge"        (Type.fun_ty nat (Type.fun_ty nat Type.bool_ty))

let failed = ref 0
let passed = ref 0

let check name cond =
  if cond then begin incr passed; Printf.printf "  PASS  %s\n" name end
  else begin
    incr failed;
    Printf.printf "  FAIL  %s\n" name
  end

(* --- Build the corpus ------------------------------------------------ *)

let euclid_corpus = [
  ("euclid", Provers.Scripted.euclid_cert ())
]

let synth_corpus =
  List.mapi (fun i c -> (Printf.sprintf "synth-%d" i, c))
    (Synth.gen ~seed:0 ~n:200 ())

let tiny_corpus =
  let nat = Type.Tyapp ("nat", []) in
  let () = if not (Type.well_formed nat) then Type.register_tyconstr "nat" 0 in
  let x = Term.Var ("x", nat) in
  [("refl-toy", {
    Cert.steps = [{
      Cert.id = 1; rule = "REFL"; witness = Cert.W_term x;
      premises = []; declared_concl = None
    }];
    concl = Term.mk_eq x x
  })]

let corpus = euclid_corpus @ tiny_corpus @ synth_corpus

(* --- Round-trip ----------------------------------------------------- *)

let rec cert_alpha_eq (a : Cert.t) (b : Cert.t) =
  Term.alpha_eq a.concl b.concl &&
  List.length a.steps = List.length b.steps &&
  List.for_all2 step_alpha_eq a.steps b.steps

and step_alpha_eq (a : Cert.step) (b : Cert.step) =
  a.id = b.id && a.rule = b.rule && a.premises = b.premises &&
  witness_alpha_eq a.witness b.witness

and witness_alpha_eq (a : Cert.witness) (b : Cert.witness) =
  match a, b with
  | Cert.W_none, Cert.W_none -> true
  | Cert.W_term t1, Cert.W_term t2 -> Term.alpha_eq t1 t2
  | Cert.W_type t1, Cert.W_type t2 -> Type.equal t1 t2
  | Cert.W_var (_, ty1), Cert.W_var (_, ty2) -> Type.equal ty1 ty2
  | Cert.W_axiom n1, Cert.W_axiom n2 -> n1 = n2
  | Cert.W_inst l1, Cert.W_inst l2 ->
    List.length l1 = List.length l2 &&
    List.for_all2 (fun (v1, t1) (v2, t2) ->
      Term.alpha_eq v1 v2 && Term.alpha_eq t1 t2) l1 l2
  | Cert.W_inst_type l1, Cert.W_inst_type l2 ->
    List.length l1 = List.length l2 &&
    List.for_all2 (fun (_, ty1) (_, ty2) -> Type.equal ty1 ty2) l1 l2
  | Cert.W_bound_and_witness ((_, ty1), w1),
    Cert.W_bound_and_witness ((_, ty2), w2) ->
    Type.equal ty1 ty2 && Term.alpha_eq w1 w2
  | _ -> false

let test_roundtrip () =
  Printf.printf "Round-trip:\n";
  let first_fail = ref true in
  List.iter (fun (name, c) ->
    let toks, hdr = Encode.cert c in
    let c' =
      try Decode.cert hdr toks
      with e -> Printf.printf "  decode error in %s: %s\n" name (Printexc.to_string e);
                { Cert.steps = []; concl = c.concl }
    in
    let ok = cert_alpha_eq c c' in
    if (not ok) && !first_fail then begin
      first_fail := false;
      Printf.printf "  -- first failure (%s) --\n" name;
      Printf.printf "    orig.concl : %s\n" (Term.to_string c.concl);
      Printf.printf "    dec.concl  : %s\n" (Term.to_string c'.concl);
      Printf.printf "    n_steps orig=%d dec=%d\n"
        (List.length c.steps) (List.length c'.steps);
      List.iter2 (fun (s : Cert.step) (s' : Cert.step) ->
        Printf.printf "      step %d %s -> dec rule=%s, witness=%s\n"
          s.id s.rule s'.rule
          (match s.witness, s'.witness with
           | Cert.W_term t1, Cert.W_term t2 ->
             Printf.sprintf "term: %s | %s"
               (Term.to_string t1) (Term.to_string t2)
           | Cert.W_var ((_, _)), Cert.W_var ((_, _)) -> "var"
           | Cert.W_none, Cert.W_none -> "none"
           | _ -> "(mismatch)")
      ) c.steps (List.filter (fun (s : Cert.step) ->
        List.exists (fun (s2 : Cert.step) -> s2.id = s.id) c.steps) c'.steps)
    end;
    check (Printf.sprintf "roundtrip %s" name) ok
  ) corpus

(* --- Grammar accept -------------------------------------------------- *)

let test_grammar () =
  Printf.printf "\nGrammar acceptance:\n";
  List.iter (fun (name, c) ->
    let toks, _ = Encode.cert c in
    let state = ref Grammar.initial in
    let ok = ref true in
    let pos = ref 0 in
    (try
       Array.iter (fun t ->
         match Grammar.step !state t with
         | Some s' -> state := s'; incr pos
         | None ->
           ok := false;
           Printf.printf "  rejected at pos %d (tok %s) in %s\n"
             !pos (Lexicon.to_string t) name;
           raise Exit
       ) toks
     with Exit -> ());
    if !ok then begin
      if not (Grammar.is_accepting !state) then begin
        Printf.printf "  not in accepting state after %s\n" name;
        ok := false
      end
    end;
    check (Printf.sprintf "grammar accepts %s" name) !ok
  ) corpus

(* --- Mask soundness -------------------------------------------------- *)

let test_mask () =
  Printf.printf "\nMask soundness:\n";
  List.iter (fun (name, c) ->
    let toks, _ = Encode.cert c in
    let state = ref Grammar.initial in
    let ok = ref true in
    let pos = ref 0 in
    (try
       Array.iter (fun t ->
         let mask = Grammar.valid_next_mask !state in
         if not mask.(t) then begin
           ok := false;
           Printf.printf "  mask excludes legal tok %s at pos %d in %s\n"
             (Lexicon.to_string t) !pos name;
           raise Exit
         end;
         (match Grammar.step !state t with
          | Some s' -> state := s'; incr pos
          | None ->
            ok := false;
            Printf.printf "  grammar rejected at pos %d in %s\n" !pos name;
            raise Exit)
       ) toks
     with Exit -> ());
    check (Printf.sprintf "mask sound %s" name) !ok
  ) corpus

(* --- Smoke: vocab_size ----------------------------------------------- *)

let test_vocab () =
  Printf.printf "Vocab:\n";
  check "vocab_size <= 256" (Lexicon.vocab_size <= 256);
  check "BOS = 0" (Lexicon.bos = 0);
  let m = Grammar.valid_next_mask Grammar.initial in
  let lparen_only = ref true in
  for i = 0 to Lexicon.vocab_size - 1 do
    if i <> Lexicon.lparen && m.(i) then lparen_only := false
  done;
  check "initial mask = {LPAREN}" (m.(Lexicon.lparen) && !lparen_only)

let () =
  test_vocab ();
  test_roundtrip ();
  test_grammar ();
  test_mask ();
  Printf.printf "\n%d passed, %d failed\n" !passed !failed;
  if !failed > 0 then exit 1

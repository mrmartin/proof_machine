(* frontend/theory.ml — load a theory package.

   A theory is a directory containing a single [theory.thy] file with
   declarations of the forms:

     type <Name>                            -- arity-0 type constructor
     type <Name> <arity>                    -- arity-N type constructor
     const <Name> : <type>                  -- constant declaration
     axiom <Name> : <term>                  -- declared axiom
     theorem <Name> : <term>                -- a theorem to be proven

   Declarations are processed in order: types before constants, both
   before axioms and theorems.

   The frontend is *untrusted*.  Type and constant registrations write
   into the kernel's tables — a malicious theory could register a
   constant with the wrong type, but the certificate verifier will
   reject any cert that uses such inconsistent shapes.  Axioms become
   trusted assumptions of the theory; the renderer lists them.  *)

type decl =
  | DType of string * int
  | DConst of string * Kernel.Type.hol_type
  | DAxiom of string * Kernel.Term.term
  | DTheorem of string * Kernel.Term.term

type theory = {
  name : string;
  decls : decl list;
}

(* ---------- Parsing ---------------------------------------------------- *)

(* The .thy file is line-oriented.  Lines starting with '#' or whitespace-
   only are skipped.  Each non-blank line is one declaration. *)

let strip s =
  let n = String.length s in
  let i = ref 0 in
  while !i < n && (s.[!i] = ' ' || s.[!i] = '\t') do incr i done;
  let j = ref (n - 1) in
  while !j >= !i && (s.[!j] = ' ' || s.[!j] = '\t' || s.[!j] = '\r')
  do decr j done;
  String.sub s !i (!j - !i + 1)

let split_at_colon s =
  match String.index_opt s ':' with
  | None -> (s, "")
  | Some i -> (strip (String.sub s 0 i),
               strip (String.sub s (i + 1) (String.length s - i - 1)))

let parse_decl line =
  let line = strip line in
  if line = "" || line.[0] = '#' then None
  else
    let words = String.split_on_char ' ' line in
    let words = List.filter (fun s -> s <> "") words in
    match words with
    | [] -> None
    | "type" :: name :: rest ->
        let arity = match rest with
          | [] -> 0
          | [n] -> int_of_string n
          | _ -> failwith ("bad type decl: " ^ line)
        in
        Some (DType (name, arity))
    | "const" :: rest ->
        let body = String.concat " " rest in
        let (name, ty_str) = split_at_colon body in
        Some (DConst (name, Kernel.Cert.parse_type ty_str))
    | "axiom" :: rest ->
        let body = String.concat " " rest in
        let (name, term_str) = split_at_colon body in
        Some (DAxiom (name, Kernel.Cert.parse_term term_str))
    | "theorem" :: rest ->
        let body = String.concat " " rest in
        let (name, term_str) = split_at_colon body in
        Some (DTheorem (name, Kernel.Cert.parse_term term_str))
    | _ -> failwith ("unrecognised .thy line: " ^ line)

(* Read a .thy file and register the types and constants (so subsequent
   term parses see them), returning the full decl list. *)
let load_file ?(name="<anonymous>") path =
  let ic = open_in path in
  let lines = ref [] in
  (try while true do lines := input_line ic :: !lines done
   with End_of_file -> ());
  close_in ic;
  let raw = List.rev !lines in
  (* Two-pass: first register types and constants, then parse axioms/
     theorems (which may reference those types and constants). *)
  let raw_decls = List.filter_map (fun l ->
    let l = strip l in
    if l = "" || l.[0] = '#' then None else Some l) raw in
  (* Pass 1: types and constants. *)
  List.iter (fun line ->
    let words = String.split_on_char ' ' line in
    let words = List.filter (fun s -> s <> "") words in
    match words with
    | "type" :: name :: rest ->
        let arity = match rest with
          | [] -> 0
          | [n] -> int_of_string n
          | _ -> failwith ("bad type decl: " ^ line)
        in
        Kernel.Type.register_tyconstr name arity
    | "const" :: rest ->
        let body = String.concat " " rest in
        let (n, ty_str) = split_at_colon body in
        let ty = Kernel.Cert.parse_type ty_str in
        Kernel.Term.register_const n ty
    | _ -> ()
  ) raw_decls;
  (* Pass 2: full decl list. *)
  let decls = List.filter_map parse_decl raw_decls in
  (* Register axioms with the kernel manifest. *)
  List.iter (function
    | DAxiom (n, t) -> Kernel.Axioms.declare n t
    | _ -> ()) decls;
  { name; decls }

let theorems thy =
  List.filter_map (function
    | DTheorem (n, t) -> Some (n, t)
    | _ -> None) thy.decls

let axioms thy =
  List.filter_map (function
    | DAxiom (n, t) -> Some (n, t)
    | _ -> None) thy.decls

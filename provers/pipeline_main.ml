(* provers/pipeline_main.ml — the `prove` CLI.

   Usage:
     prove --using <name1,name2,...> <phi.kf> <out.cert>

   Reads φ (and any declared axioms) from the .kf file, runs each
   prover in turn, verifies the first emitted certificate against φ,
   and writes it to <out.cert> on success. *)

let registered = [
  ("lookup",     fun ~phi ~budget ~hints ->
     Provers.Lookup.prove ~phi ~budget ~hints);
  ("scripted",   fun ~phi ~budget ~hints ->
     Provers.Scripted.prove ~phi ~budget ~hints);
  ("enumerator", fun ~phi ~budget ~hints ->
     Provers.Enumerator.prove ~phi ~budget ~hints);
]

let usage () =
  prerr_endline
    "usage: prove --using <p1,p2,...> <phi.kf> <out.cert>";
  exit 2

(* --- Write a certificate to a .cert file in S-expression form. -------- *)

let escape s =
  let buf = Buffer.create (String.length s + 4) in
  String.iter (fun c ->
    match c with
    | '"' -> Buffer.add_string buf "\\\""
    | '\\' -> Buffer.add_string buf "\\\\"
    | '\n' -> Buffer.add_string buf "\\n"
    | c -> Buffer.add_char buf c) s;
  Buffer.contents buf

let pp_type = Kernel.Type.to_string
let pp_term = Kernel.Cert.pp_term

let write_witness oc = function
  | Kernel.Cert.W_none ->
      Printf.fprintf oc "(witness ())"
  | W_term t ->
      Printf.fprintf oc "(witness (term \"%s\"))" (escape (pp_term t))
  | W_type ty ->
      Printf.fprintf oc "(witness (type \"%s\"))" (escape (pp_type ty))
  | W_var (n, ty) ->
      Printf.fprintf oc "(witness (var \"%s\" \"%s\"))"
        (escape n) (escape (pp_type ty))
  | W_axiom n ->
      Printf.fprintf oc "(witness (axiom \"%s\"))" (escape n)
  | W_bound_and_witness ((n, ty), w) ->
      Printf.fprintf oc
        "(witness (bound_and_witness (bound \"%s\" \"%s\") (witness \"%s\")))"
        (escape n) (escape (pp_type ty)) (escape (pp_term w))
  | W_inst _ | W_inst_type _ ->
      Printf.fprintf oc "(witness ())" (* not used by current MVP provers *)

let write_step oc (s : Kernel.Cert.step) =
  Printf.fprintf oc "  (step %d (rule %s) " s.id s.rule;
  write_witness oc s.witness;
  Printf.fprintf oc " (premises";
  List.iter (fun i -> Printf.fprintf oc " %d" i) s.premises;
  Printf.fprintf oc "))\n"

let write_cert (cert : Kernel.Cert.t) out_path =
  let oc = open_out out_path in
  Printf.fprintf oc "(cert\n";
  List.iter (write_step oc) cert.steps;
  Printf.fprintf oc "  (concl \"%s\"))\n" (escape (pp_term cert.concl));
  close_out oc

(* --- Main --------------------------------------------------------------- *)

let () =
  let argv = Array.to_list Sys.argv in
  let usings = ref ["lookup"; "scripted"; "enumerator"] in
  let positional = ref [] in
  let rec parse = function
    | [] -> ()
    | "--using" :: v :: rest ->
        usings := String.split_on_char ',' v;
        parse rest
    | x :: rest ->
        positional := x :: !positional;
        parse rest
  in
  parse (List.tl argv);
  let positional = List.rev !positional in
  let (phi_path, out_path) = match positional with
    | [p; o] -> (p, o)
    | _ -> usage ()
  in
  let (axioms, goal_opt) = Kernel.Cert.parse_phi_file phi_path in
  List.iter (fun (n, t) -> Kernel.Axioms.declare n t) axioms;
  let phi = match goal_opt with
    | Some g -> g
    | None -> prerr_endline "prove: no goal in .kf file"; exit 2
  in
  let try_prover name =
    match List.assoc_opt name registered with
    | None ->
        Printf.eprintf "prove: unknown prover '%s'\n" name; None
    | Some f ->
        let certs = f ~phi ~budget:Provers.Prover_api.default_budget ~hints:[] in
        let rec pick s =
          match s () with
          | Seq.Nil -> None
          | Seq.Cons (c, rest) ->
              (match Kernel.Verify.verify c phi with
               | Kernel.Verify.Ok -> Some (name, c)
               | _ -> pick rest)
        in
        pick certs
  in
  let rec run = function
    | [] -> None
    | n :: rest ->
        (match try_prover n with
         | Some r -> Some r
         | None -> run rest)
  in
  match run !usings with
  | None ->
      prerr_endline "prove: no prover produced an accepted certificate";
      exit 1
  | Some (used, cert) ->
      write_cert cert out_path;
      (* Also seed the lookup cache. *)
      Provers.Lookup.store phi out_path;
      Printf.printf "prove: %s emitted certificate -> %s\n" used out_path

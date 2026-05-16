(* frontend/elab_main.ml — the `elab` CLI.

   Usage:
     elab <theory.thy> <theorem-name> <out.kf>

   Loads the theory, picks the named theorem, writes a .kf file that
   declares the theory's axioms and states the theorem's φ. *)

let () =
  if Array.length Sys.argv < 4 then begin
    prerr_endline "usage: elab <theory.thy> <theorem-name> <out.kf>";
    exit 2
  end;
  let theory_path = Sys.argv.(1) in
  let theorem_name = Sys.argv.(2) in
  let out_path = Sys.argv.(3) in
  let thy = Frontend.Theory.load_file theory_path in
  Frontend.Elaborate.elaborate_theorem thy theorem_name out_path;
  Printf.printf "elab: %s -> %s\n" theorem_name out_path

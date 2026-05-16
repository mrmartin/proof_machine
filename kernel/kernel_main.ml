(* kernel_main.ml — the `vrfy` CLI.

   Usage:  vrfy  <phi.kf>  <cert.cert>

   The .kf file declares any theory-package axioms used by the
   certificate and states the goal; the .cert file is the
   certificate to verify.  Exits 0 on accept, 1 on reject. *)

let () =
  if Array.length Sys.argv < 3 then begin
    prerr_endline "usage: vrfy <phi.kf> <cert.cert>";
    exit 2
  end;
  let phi_path = Sys.argv.(1) in
  let cert_path = Sys.argv.(2) in
  let (axioms, goal_opt) = Kernel.Cert.parse_phi_file phi_path in
  List.iter (fun (n, t) -> Kernel.Axioms.declare n t) axioms;
  match goal_opt with
  | None ->
      prerr_endline "vrfy: no (goal ...) form in phi file";
      exit 2
  | Some phi ->
      let cert = Kernel.Cert.parse_file cert_path in
      match Kernel.Verify.verify cert phi with
      | Kernel.Verify.Ok ->
          print_endline "accept";
          exit 0
      | Kernel.Verify.Reject msg ->
          prerr_endline ("vrfy: REJECT — " ^ msg);
          exit 1

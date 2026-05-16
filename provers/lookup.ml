(* provers/lookup.ml — disk-cached certificate lookup.

   Key: a printable hash of the canonical-form term.  Value: the
   cached certificate file.  Cache lives in
   [~/.proof_machine/cache] by default. *)

let cache_dir () =
  let home = try Sys.getenv "HOME" with Not_found -> "." in
  Filename.concat home ".proof_machine/cache"

let ensure_dir d =
  ignore (Sys.command (Printf.sprintf "mkdir -p %s" (Filename.quote d)))

let canonical phi = Kernel.Cert.pp_term phi

let key phi =
  let s = canonical phi in
  Printf.sprintf "%016x" (Hashtbl.hash s)

let store phi cert_path =
  let d = cache_dir () in
  ensure_dir d;
  let dst = Filename.concat d (key phi ^ ".cert") in
  let ic = open_in cert_path in
  let oc = open_out dst in
  let n = in_channel_length ic in
  let buf = Bytes.create n in
  really_input ic buf 0 n;
  output_bytes oc buf;
  close_in ic;
  close_out oc

let name = "lookup"

let prove ~phi ~budget:_ ~hints:_ =
  let path = Filename.concat (cache_dir ()) (key phi ^ ".cert") in
  if Sys.file_exists path then
    try
      let cert = Kernel.Cert.parse_file path in
      Seq.return cert
    with _ -> Seq.empty
  else Seq.empty

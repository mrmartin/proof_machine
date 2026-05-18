(* kernel/verify.ml — the trusted verifier V.

   V(π, φ) replays every step in π by calling the named primitive rule
   on the already-checked premise theorems.  Accepts iff every rule
   succeeds and the final step's conclusion is alpha-equivalent to φ. *)

type result = Ok | Reject of string

let verify (cert : Cert.t) (phi : Term.term) : result =
  if cert.steps = [] then Reject "empty certificate" else
  let table : (int, Thm.t) Hashtbl.t = Hashtbl.create 64 in
  let last_id = ref 0 in
  let rec loop = function
    | [] -> Ok
    | (step : Cert.step) :: rest ->
        let outcome =
          match Cert.apply_step table step with
          | thm ->
              (match step.declared_concl with
               | Some d when not (Term.alpha_eq d (Thm.concl thm)) ->
                   `Bad (Printf.sprintf
                           "step %d: declared concl %s ≠ derived %s"
                           step.id (Term.to_string d)
                           (Term.to_string (Thm.concl thm)))
               | _ ->
                   Hashtbl.add table step.id thm;
                   last_id := step.id;
                   `Good)
          | exception Rules.Rule_error m ->
              `Bad (Printf.sprintf "step %d (%s): rule rejected: %s"
                      step.id step.rule m)
          | exception Failure m ->
              `Bad (Printf.sprintf "step %d (%s): failure: %s"
                      step.id step.rule m)
        in
        match outcome with
        | `Good -> loop rest
        | `Bad msg -> Reject msg
  in
  match loop cert.steps with
  | Reject _ as r -> r
  | Ok ->
      let final = Hashtbl.find table !last_id in
      let derived = Thm.concl final in
      if not (Term.alpha_eq derived cert.concl) then
        Reject (Printf.sprintf
                  "final derived %s ≠ cert's declared concl %s"
                  (Term.to_string derived)
                  (Term.to_string cert.concl))
      else if not (Term.alpha_eq cert.concl phi) then
        Reject (Printf.sprintf
                  "cert's concl %s ≠ stated goal %s"
                  (Term.to_string cert.concl)
                  (Term.to_string phi))
      else Ok

let to_bool = function Ok -> true | _ -> false

let reason = function Ok -> "accept" | Reject s -> "reject: " ^ s

(* --- Prefix verification ----------------------------------------------
   Applies the first [k] steps of a cert through the kernel and returns
   the derived theorems (paired with their declared step IDs) in step
   order.  Used by [verify_tokens]'s prefix-mode protocol — both the
   tree-search driver (M1) and the proof-state exposure path (M3) need
   to read the theorem table after a partial proof.

   Crucially, this uses the same primitive bindings as [verify]; the
   readback is a serialisation of state the verifier already maintains
   internally (docs/CERTIFICATE.md:9–21).  No new trust surface. *)

type prefix_result =
  | Prefix_ok of (int * Thm.t) list  (* in step order *)
  | Prefix_reject of int * string    (* failing step index, message *)

let verify_prefix (cert : Cert.t) (k : int) : prefix_result =
  if k < 0 then Prefix_reject (-1, "prefix length negative") else
  let table : (int, Thm.t) Hashtbl.t = Hashtbl.create 64 in
  let derived = ref [] in
  let rec loop i = function
    | _ when i >= k -> Prefix_ok (List.rev !derived)
    | [] -> Prefix_ok (List.rev !derived)
    | (step : Cert.step) :: rest ->
        (match Cert.apply_step table step with
         | thm ->
             (match step.declared_concl with
              | Some d when not (Term.alpha_eq d (Thm.concl thm)) ->
                  Prefix_reject (i,
                    Printf.sprintf
                      "step %d: declared concl %s ≠ derived %s"
                      step.id (Term.to_string d)
                      (Term.to_string (Thm.concl thm)))
              | _ ->
                  Hashtbl.add table step.id thm;
                  derived := (step.id, thm) :: !derived;
                  loop (i + 1) rest)
         | exception Rules.Rule_error m ->
             Prefix_reject (i,
               Printf.sprintf "step %d (%s): rule rejected: %s"
                 step.id step.rule m)
         | exception Failure m ->
             Prefix_reject (i,
               Printf.sprintf "step %d (%s): failure: %s"
                 step.id step.rule m))
  in
  loop 0 cert.steps

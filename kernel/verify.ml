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

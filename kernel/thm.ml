(* kernel/thm.ml — the opaque theorem type.

   A [Thm.t] represents a sequent  hypotheses ⊢ conclusion  in HOL.

   Only the kernel library may construct [Thm.t] values; outside code
   sees the type as abstract (see [kernel.ml], which is the library's
   umbrella interface and omits [mk]).  The kernel's own primitive
   rules and axioms call [mk] to mint theorems; nothing else may. *)

type t = { hyps : Term.term list; concl : Term.term }

let mk hyps concl = { hyps; concl }
let concl t = t.concl
let hyps t = t.hyps

(* Union of hypothesis lists, deduplicated up to alpha-equivalence.  HOL
   sequents track hypotheses as sets, not multisets. *)
let union_hyps xs ys =
  List.fold_left
    (fun acc h ->
       if List.exists (Term.alpha_eq h) acc then acc else h :: acc)
    xs ys

(* Remove from xs anything alpha-equivalent to anything in ys. *)
let remove_hyps xs ys =
  List.filter (fun h -> not (List.exists (Term.alpha_eq h) ys)) xs

let to_string th =
  let hyps_str = match th.hyps with
    | [] -> ""
    | hs -> String.concat ", " (List.map Term.to_string hs) ^ " "
  in
  hyps_str ^ "|- " ^ Term.to_string th.concl

(* provers/prover_api.ml — the plugin contract for prover backends.

   Any module implementing PROVER may be added to the pipeline; the
   kernel re-checks every certificate the prover emits.  Untrusted by
   design. *)

type budget = {
  max_steps : int;
  max_seconds : float;
}

let default_budget = { max_steps = 1_000_000; max_seconds = 30.0 }

type hint = string

module type PROVER = sig
  val name : string
  val prove :
    phi:Kernel.Term.term ->
    budget:budget ->
    hints:hint list ->
    Kernel.Cert.t Seq.t
end

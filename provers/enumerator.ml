(* provers/enumerator.ml — bounded Hilbert enumeration.

   Pedagogical baseline.  Currently emits one-step certificates of the
   form  ⊢ t = t  via REFL when the goal has that shape.  A real
   enumerator would BFS through axiom instantiations and rule
   applications up to a depth bound; for the MVP we keep it minimal —
   the architectural point is that an enumerator plugs into the same
   socket as any other prover. *)

let name = "enumerator"

let prove ~phi ~budget:_ ~hints:_ =
  match phi with
  | Kernel.Term.Comb (Kernel.Term.Comb (Kernel.Term.Const ("=", _), s), t)
    when Kernel.Term.alpha_eq s t ->
      let cert = Kernel.Cert.{
        steps = [{
          id = 1;
          rule = "REFL";
          witness = W_term s;
          premises = [];
          declared_concl = None;
        }];
        concl = phi;
      } in
      Seq.return cert
  | _ -> Seq.empty

(* render/render_main.ml — turn a (.kf, .cert) pair into a LaTeX
   document with the theorem statement, declared axioms, and a
   structured proof trace.

   Usage:  render <phi.kf>  <cert.cert>  <out.tex> *)

(* --- LaTeX pretty-printer for HOL terms ----------------------------- *)

let rec latex_of_term t =
  let open Kernel.Term in
  match t with
  (* Higher-order patterns first. *)
  | Comb (Const ("!", _), Abs (Var (n, ty), body)) ->
      Printf.sprintf "\\forall %s\\!:\\!%s.\\, %s"
        (latex_name n) (latex_of_type ty) (latex_of_term body)
  | Comb (Const ("?", _), Abs (Var (n, ty), body)) ->
      Printf.sprintf "\\exists %s\\!:\\!%s.\\, %s"
        (latex_name n) (latex_of_type ty) (latex_of_term body)
  | Comb (Comb (Const ("=", _), a), b) ->
      Printf.sprintf "%s = %s" (latex_of_term a) (latex_of_term b)
  | Comb (Comb (Const ("/\\", _), a), b) ->
      Printf.sprintf "(%s \\wedge %s)" (latex_of_term a) (latex_of_term b)
  | Comb (Comb (Const ("\\/", _), a), b) ->
      Printf.sprintf "(%s \\vee %s)" (latex_of_term a) (latex_of_term b)
  | Comb (Comb (Const ("==>", _), a), b) ->
      Printf.sprintf "(%s \\Rightarrow %s)" (latex_of_term a) (latex_of_term b)
  | Comb (Const ("~", _), a) ->
      Printf.sprintf "\\neg %s" (latex_of_term a)
  (* Number-theory domain operators. *)
  | Comb (Comb (Const ("plus", _), a), b) ->
      Printf.sprintf "(%s + %s)" (latex_of_term a) (latex_of_term b)
  | Comb (Const ("factorial", _), a) ->
      Printf.sprintf "%s!" (latex_of_term a)
  | Comb (Const ("prime", _), a) ->
      Printf.sprintf "\\mathrm{prime}(%s)" (latex_of_term a)
  | Comb (Comb (Const ("divides", _), a), b) ->
      Printf.sprintf "(%s \\mid %s)" (latex_of_term a) (latex_of_term b)
  | Comb (Comb (Const ("gt", _), a), b) ->
      Printf.sprintf "(%s > %s)" (latex_of_term a) (latex_of_term b)
  | Comb (Comb (Const ("ge", _), a), b) ->
      Printf.sprintf "(%s \\geq %s)" (latex_of_term a) (latex_of_term b)
  (* Generic forms. *)
  | Var (n, _) -> latex_name n
  | Const (c, _) -> latex_const c
  | Comb (f, x) ->
      Printf.sprintf "%s\\,%s" (latex_of_term f) (latex_of_term x)
  | Abs (Var (n, ty), b) ->
      Printf.sprintf "\\lambda %s\\!:\\!%s.\\, %s"
        (latex_name n) (latex_of_type ty) (latex_of_term b)
  | Abs _ -> "\\langle bad abs\\rangle"

and latex_name n = escape_underscores n
and latex_const c =
  match c with
  | "one" -> "1"
  | "T" -> "\\top"
  | "F" -> "\\bot"
  | _ -> "\\mathrm{" ^ escape_underscores c ^ "}"

and escape_underscores s =
  let buf = Buffer.create (String.length s) in
  String.iter (fun c ->
    if c = '_' then Buffer.add_string buf "\\_"
    else Buffer.add_char buf c) s;
  Buffer.contents buf

and escape_text s = escape_underscores s
and latex_of_type ty =
  let open Kernel.Type in
  match ty with
  | Tyapp ("nat", []) -> "\\mathbb{N}"
  | Tyapp ("bool", []) -> "\\mathbb{B}"
  | Tyapp ("ind", []) -> "\\iota"
  | Tyapp ("fun", [a; b]) ->
      Printf.sprintf "(%s \\to %s)" (latex_of_type a) (latex_of_type b)
  | Tyvar a -> "\\alpha_{" ^ a ^ "}"
  | Tyapp (s, []) -> "\\mathrm{" ^ s ^ "}"
  | Tyapp (s, args) ->
      Printf.sprintf "\\mathrm{%s}(%s)" s
        (String.concat ", " (List.map latex_of_type args))

(* --- A textual rule description for the trace column ----------------- *)

let rule_label = function
  | "REFL" -> "reflexivity"
  | "TRANS" -> "transitivity"
  | "MK_COMB" -> "congruence (MK\\_COMB)"
  | "ABS" -> "abstraction"
  | "BETA" -> "$\\beta$-reduction"
  | "ASSUME" -> "assume"
  | "EQ_MP" -> "EQ\\_MP"
  | "DEDUCT_ANTISYM_RULE" -> "deduction (antisym)"
  | "INST" -> "instantiation"
  | "INST_TYPE" -> "type instantiation"
  | "GEN" -> "$\\forall$-intro (GEN)"
  | "SPEC" -> "$\\forall$-elim (SPEC)"
  | "EXISTS" -> "$\\exists$-intro (EXISTS)"
  | "CHOOSE" -> "$\\exists$-elim (CHOOSE)"
  | "CONJ" -> "$\\wedge$-intro"
  | "CONJUNCT1" -> "$\\wedge$-elim left"
  | "CONJUNCT2" -> "$\\wedge$-elim right"
  | "MP" -> "modus ponens"
  | "DISCH" -> "deduction (DISCH)"
  | "AXIOM" -> "\\textbf{axiom}"
  | "ETA_AX" -> "\\textbf{axiom} ETA"
  | "SELECT_AX" -> "\\textbf{axiom} SELECT"
  | "EM_AX" -> "\\textbf{axiom} EM"
  | s -> s

let witness_str = function
  | Kernel.Cert.W_none -> ""
  | W_term t -> "at $" ^ latex_of_term t ^ "$"
  | W_type ty -> "at $" ^ latex_of_type ty ^ "$"
  | W_var (n, ty) -> "with $" ^ latex_name n ^ "\\!:\\!" ^ latex_of_type ty ^ "$"
  | W_axiom n -> "[\\texttt{" ^ escape_text n ^ "}]"
  | W_bound_and_witness ((n, _), w) ->
      "binding $" ^ latex_name n ^ "$ at $" ^ latex_of_term w ^ "$"
  | W_inst _ | W_inst_type _ -> ""

let premises_str = function
  | [] -> "--"
  | xs -> String.concat ", " (List.map string_of_int xs)

(* --- A simple step-by-step replay to get each step's derived thm so
       we can print its conclusion in the trace. --------------------- *)

let replay (cert : Kernel.Cert.t) =
  let table = Hashtbl.create 64 in
  let results = ref [] in
  List.iter (fun (s : Kernel.Cert.step) ->
    let thm = try Kernel.Cert.apply_step table s
              with e -> failwith ("render: replay failure at step "
                                  ^ string_of_int s.id
                                  ^ ": " ^ Printexc.to_string e)
    in
    Hashtbl.add table s.id thm;
    results := (s, thm) :: !results
  ) cert.steps;
  List.rev !results

(* --- Document writer ------------------------------------------------- *)

let header thm_name = Printf.sprintf {|\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{longtable}
\usepackage{xcolor}
\newtheorem{theorem}{Theorem}
\newcommand{\stepid}[1]{{\color{gray}\scriptsize #1}}
\title{%s}
\author{\texttt{proof\_machine}}
\date{Verified by the sealed kernel V}
\begin{document}
\maketitle
|} thm_name

let footer = "\\end{document}\n"

let render ~thm_name ~axioms ~goal ~cert oc =
  output_string oc (header thm_name);
  Printf.fprintf oc "\\section*{Statement}\n";
  Printf.fprintf oc "\\begin{theorem}[%s]\n$$%s$$\n\\end{theorem}\n\n"
    thm_name (latex_of_term goal);
  if axioms <> [] then begin
    Printf.fprintf oc "\\section*{Declared axioms}\n";
    Printf.fprintf oc "The proof cites the following theory-package axioms (trusted assumptions for v0.1):\n";
    Printf.fprintf oc "\\begin{itemize}\n";
    List.iter (fun (n, t) ->
      Printf.fprintf oc "  \\item \\texttt{%s}: $%s$\n"
        (escape_text n) (latex_of_term t)
    ) axioms;
    Printf.fprintf oc "\\end{itemize}\n\n"
  end;
  Printf.fprintf oc "\\section*{Proof trace}\n";
  Printf.fprintf oc "Each row is one primitive kernel step. The trace was verified by \\texttt{vrfy} as a Cook--Reckhow certificate.\n\n";
  Printf.fprintf oc "\\begin{longtable}{rlp{6.5cm}l}\n";
  Printf.fprintf oc "\\textbf{\\#} & \\textbf{rule} & \\textbf{conclusion} & \\textbf{from} \\\\\n";
  Printf.fprintf oc "\\hline\n";
  let trace = replay cert in
  List.iter (fun ((s : Kernel.Cert.step), thm) ->
    let w = witness_str s.witness in
    let wsep = if w = "" then "" else " " ^ w in
    Printf.fprintf oc "\\stepid{%d} & %s%s & $%s$ & %s \\\\\n"
      s.id (rule_label s.rule) wsep
      (latex_of_term (Kernel.Thm.concl thm))
      (premises_str s.premises)
  ) trace;
  Printf.fprintf oc "\\end{longtable}\n\n";
  Printf.fprintf oc "\\section*{Provenance}\n";
  Printf.fprintf oc "The trusted base for this document is the kernel verifier $V$ (see \\texttt{kernel/verify.ml}) plus the axioms listed above.\n";
  output_string oc footer

let () =
  if Array.length Sys.argv < 4 then begin
    prerr_endline "usage: render <phi.kf> <cert.cert> <out.tex>";
    exit 2
  end;
  let phi_path = Sys.argv.(1) in
  let cert_path = Sys.argv.(2) in
  let out_path = Sys.argv.(3) in
  let (axioms, goal_opt) = Kernel.Cert.parse_phi_file phi_path in
  List.iter (fun (n, t) -> Kernel.Axioms.declare n t) axioms;
  let goal = match goal_opt with
    | Some g -> g
    | None -> prerr_endline "render: no goal"; exit 2
  in
  let cert = Kernel.Cert.parse_file cert_path in
  let thm_name =
    Filename.basename phi_path
    |> Filename.remove_extension
  in
  let oc = open_out out_path in
  render ~thm_name ~axioms ~goal ~cert oc;
  close_out oc;
  Printf.printf "render: %s -> %s\n" thm_name out_path

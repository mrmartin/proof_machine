# Number Theory — Theory package for proof_machine.
#
# Declares the type and constants needed to state Euclid's infinitude
# of primes, the helper axioms used by the proof, and the headline
# theorem itself.
#
# In v0.1 the helper lemmas are *declared as axioms*.  In v0.2 each
# would be replaced by a verified certificate.  The renderer marks
# axioms as such in the output, so trust is explicit.

# --- Type and constants ---------------------------------------------------

type nat

const one        : nat
const factorial  : nat -> nat
const plus       : nat -> nat -> nat
const prime      : nat -> bool
const divides    : nat -> nat -> bool
const gt         : nat -> nat -> bool
const ge         : nat -> nat -> bool

# --- Declared axioms (helper lemmas) -------------------------------------

axiom fact_pos              : (! n : nat. (ge (plus (factorial n) one) one))

axiom has_prime_divisor     : (! m : nat. (? p : nat. ((prime p) /\ (divides p m))))

# The substantive content of Euclid's argument, declared as a single
# axiom for v0.1.  In v0.2 it would be derived from divides_factorial_le,
# divides_diff and not_divides_one.
axiom euclid_step           : (! p : nat. (! n : nat. (((prime p) /\ (divides p (plus (factorial n) one))) ==> (gt p n))))

# --- The theorem to prove ------------------------------------------------

theorem euclid : (! n : nat. (? p : nat. ((gt p n) /\ (prime p))))

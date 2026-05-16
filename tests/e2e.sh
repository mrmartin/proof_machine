#!/usr/bin/env bash
# End-to-end test for proof_machine.
#
# Runs the full pipeline (elaborate → prove → verify → render) on:
#   1.  examples/tiny/refl.thy        — provable by the enumerator
#   2.  theories/number_theory/...    — Euclid, provable by scripted
#
# Then verifies that every adversarial certificate in tests/adversarial/
# is rejected by V.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD=_build/default
ELAB="$BUILD/frontend/elab_main.exe"
PROVE="$BUILD/provers/pipeline_main.exe"
VRFY="$BUILD/kernel/kernel_main.exe"
RENDER="$BUILD/render/render_main.exe"
TMP="$(mktemp -d)"
trap "rm -rf $TMP" EXIT

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }

ok=0
fail=0

note() { echo; echo "==> $*"; }
pass() { green "  OK  $*"; ok=$((ok+1)); }
die()  { red   "  FAIL $*"; fail=$((fail+1)); }

# -- 1. Tiny REFL example -------------------------------------------------
note "tiny/refl"
$ELAB examples/tiny/refl.thy refl "$TMP/refl.kf" > /dev/null
$PROVE --using enumerator "$TMP/refl.kf" "$TMP/refl.cert" > /dev/null
if $VRFY "$TMP/refl.kf" "$TMP/refl.cert" > /dev/null; then
  pass "tiny/refl verified"
else
  die "tiny/refl rejected"
fi

# -- 2. Euclid full pipeline ----------------------------------------------
note "euclid (number_theory)"
$ELAB theories/number_theory/theory.thy euclid "$TMP/euclid.kf" > /dev/null
$PROVE --using scripted "$TMP/euclid.kf" "$TMP/euclid.cert" > /dev/null
if $VRFY "$TMP/euclid.kf" "$TMP/euclid.cert" > /dev/null; then
  pass "euclid verified"
else
  die "euclid rejected"
fi

# -- 3. Renderer produces LaTeX -------------------------------------------
$RENDER "$TMP/euclid.kf" "$TMP/euclid.cert" "$TMP/euclid.tex" > /dev/null
if grep -q "Theorem" "$TMP/euclid.tex"; then
  pass "renderer emitted theorem statement"
else
  die "renderer output missing theorem"
fi

# -- 4. Lookup cache should now hit on Euclid -----------------------------
$PROVE --using lookup "$TMP/euclid.kf" "$TMP/euclid2.cert" > /dev/null 2>&1 \
  && pass "lookup hit after prior store" \
  || die  "lookup cache miss"

# -- 5. Adversarial certificates: V MUST reject each ----------------------
note "adversarial certs (must all be rejected)"
for f in tests/adversarial/*.cert; do
  name=$(basename "$f")
  if $VRFY tests/adversarial/phi.kf "$f" > /dev/null 2>&1; then
    die  "$name was accepted (soundness hole!)"
  else
    pass "$name rejected"
  fi
done

# -- summary --------------------------------------------------------------
echo
if [[ "$fail" -eq 0 ]]; then
  green "e2e: $ok checks passed"
  exit 0
else
  red   "e2e: $ok ok, $fail failed"
  exit 1
fi

#!/bin/sh
# Real Engine verification driver.
#
# Runs, in order:
#   1. strict clang -std=c23 build of the whole engine
#   2. GLSL validation via glslangValidator (Vulkan 1.3 SPIR-V)
#   3. headless smoke run of the showcase
#   4. CBMC bounded model checking of the proof harnesses
#   5. Frama-C/WP deductive proofs of the ACSL contracts
#   6. GenMC stateless model checking of the lock-free MPMC ring
#
# Steps 4-6 are SKIPped with an explicit message when the tool is not
# installed; every SKIP is reported so results are never silently
# invented. Exit code is non-zero only on a real failure.
set -eu

cd "$(dirname "$0")"
FAIL=0
SKIPPED=0

note()  { printf '\n=== %s ===\n' "$1"; }
pass()  { printf 'PASS  %s\n' "$1"; }
fail()  { printf 'FAIL  %s\n' "$1"; FAIL=1; }
skip()  { printf 'SKIP  %s (tool not installed)\n' "$1"; SKIPPED=1; }

note "1/6 strict build (clang -std=c23 -Wall -Wextra -Werror -pedantic -Wconversion)"
make clean all || fail "make all"

note "2/6 GLSL validation"
if command -v glslangValidator >/dev/null 2>&1; then
    mkdir -p bin/shaders
    for s in shaders/*; do
        glslangValidator -V --target-env vulkan1.3 "$s" -o "bin/shaders/$(basename "$s").spv" \
            || fail "glslangValidator $s"
    done
    pass "glslangValidator (all shaders)"
else
    skip "glslangValidator"
fi

note "3/6 headless smoke run (1000 bodies, 1 s)"
if [ -x bin/real_engine ]; then
    bin/real_engine --headless --bodies 1000 --seconds 1 || fail "headless smoke run"
    pass "headless smoke run"
else
    fail "engine binary missing"
fi

note "4/6 CBMC bounded model checking"
if command -v cbmc >/dev/null 2>&1; then
    for h in verify/*_cbmc.c; do
        [ -e "$h" ] || continue
        if cbmc "$h" --unwind 16 --unwinding-assertions --pointer-overflow-check \
             --signed-overflow-check --unsigned-overflow-check >/dev/null 2>&1; then
            pass "cbmc $(basename "$h")"
        else
            fail "cbmc $(basename "$h")"
        fi
    done
else
    skip "cbmc"
fi

note "5/6 Frama-C/WP deductive proofs"
if command -v frama-c >/dev/null 2>&1; then
    for h in verify/*_frama.c; do
        [ -e "$h" ] || continue
        if frama-c -wp -wp-rte -wp-fct main -wp-timeout 120 "$h" >/dev/null 2>&1; then
            pass "frama-c $(basename "$h")"
        else
            fail "frama-c $(basename "$h")"
        fi
    done
else
    skip "frama-c"
fi

note "6/6 GenMC stateless model checking"
if command -v genmc >/dev/null 2>&1; then
    for h in verify/*_genmc.c; do
        [ -e "$h" ] || continue
        if genmc -unroll 8 "$h" >/dev/null 2>&1; then
            pass "genmc $(basename "$h")"
        else
            fail "genmc $(basename "$h")"
        fi
    done
else
    skip "genmc"
fi

note "summary"
if [ "$FAIL" -ne 0 ]; then
    echo "verification FAILED (see FAIL lines above)"
    exit 1
fi
if [ "$SKIPPED" -ne 0 ]; then
    echo "verification passed; some tools were skipped (install cbmc/frama-c/genmc for the full suite)"
else
    echo "verification passed (full suite)"
fi
exit 0

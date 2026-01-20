#!/bin/bash
# Compare benchmark results with different compiler flags
# This script runs all benchmarks with various compiler optimization flags,
# saving each configuration to a separate baseline for comparison.
#
# Usage:
#   ./scripts/bench_compiler_flags.sh            # Run all benchmarks
#   ./scripts/bench_compiler_flags.sh "1024b"    # Filter to 1024-bit only
#   ./scripts/bench_compiler_flags.sh "512b"     # Filter to 512-bit only
#   ./scripts/bench_compiler_flags.sh "dispatch" # Filter to dispatch group only

set -e

BENCH_FILTER="${1:-}"  # Optional filter (e.g., "1024bit", "dispatch")
if [ -n "$BENCH_FILTER" ]; then
    echo "Filtering to: $BENCH_FILTER"
fi

ARCH=$(uname -m)

echo "=============================================="
echo "Compiler Flags Comparison for Hamming Distance"
echo "Architecture: $ARCH"
echo "=============================================="
echo ""

run_bench() {
    local name="$1"
    local rustflags="$2"
    local features="${3:-}"

    # Convert name to baseline-friendly format (lowercase, no spaces/dots, alphanumeric + underscores)
    local baseline
    baseline=$(echo "$name" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g' | sed 's/__*/_/g' | sed 's/^_//;s/_$//')

    echo "=== $name ==="
    echo "    (baseline: $baseline)"
    echo ""

    local cargo_args=()
    if [ -n "$features" ]; then
        cargo_args+=(--features "$features")
    fi

    local criterion_args=(--save-baseline "$baseline")
    if [ -n "$BENCH_FILTER" ]; then
        criterion_args+=("$BENCH_FILTER")
    fi

    if [ -n "$rustflags" ]; then
        RUSTFLAGS="$rustflags" cargo bench "${cargo_args[@]}" -- "${criterion_args[@]}" 2>/dev/null || echo "  [SKIPPED - CPU doesn't support required features]"
    else
        cargo bench "${cargo_args[@]}" -- "${criterion_args[@]}" || echo "  [SKIPPED - benchmark failed]"
    fi
    echo ""
}

if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
    # =========================================================================
    # ARM64 (Apple Silicon, AWS Graviton, etc.)
    # =========================================================================
    # NEON is baseline for ARM64 - always enabled automatically.
    # =========================================================================

    echo "Running ARM64 benchmarks..."
    echo ""

    echo "--- Baseline (no features) ---"
    run_bench "1a. Default" ""
    run_bench "1b. Default + native" "-C target-cpu=native"
else
    # =========================================================================
    # x86-64 (Intel/AMD)
    # =========================================================================
    # x86-64 Microarchitecture Levels:
    #   x86-64    : Baseline (SSE2 only) - maximum portability
    #   x86-64-v2 : SSE4.2, POPCNT (Nehalem 2008+)
    #   x86-64-v3 : AVX2, BMI1/2, FMA (Haswell 2013+)
    #   x86-64-v4 : AVX-512 foundation (Skylake-X 2017+)
    #   +avx512vpopcntdq : Hardware vector popcount (Ice Lake 2019+)
    # =========================================================================

    echo "Running x86-64 benchmarks..."
    echo ""

    echo "--- Baseline (no features) ---"
    run_bench "1a. Default (portable)" ""
    run_bench "1b. Default + native" "-C target-cpu=native"

    echo "--- With multiversion feature ---"
    run_bench "2. multiversion" "" "multiversion"

    echo "--- Specific x86-64 levels (no features) ---"
    run_bench "4. x86-64-v2 (SSE4.2 + POPCNT)" "-C target-cpu=x86-64-v2"
    run_bench "5. x86-64-v3 (AVX2)" "-C target-cpu=x86-64-v3"
    run_bench "6. x86-64-v4 (AVX-512)" "-C target-cpu=x86-64-v4"
    run_bench "7. x86-64-v4 + VPOPCNTDQ" "-C target-cpu=x86-64-v4 -C target-feature=+avx512vpopcntdq"
fi

echo "=============================================="
echo "Summary"
echo "=============================================="
echo ""
echo "Each configuration saved to a unique baseline in target/criterion/."
echo ""
echo "Compare baselines:"
echo "  cargo bench -- --baseline 1a_default           # Compare current vs 1a_default"
echo "  cargo bench -- --load-baseline 1a_default      # Load without running new benchmarks"
echo ""
if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
    echo "ARM64 Notes:"
    echo "  - NEON is always enabled (it's the baseline for ARM64)"
    echo "  - 'native' may enable additional features for your specific chip"
    echo "  - Performance differences are typically smaller than on x86"
else
    echo "x86-64 Notes:"
    echo "  - Default is very portable but may be slow"
    echo "  - 'native' uses all features available on this CPU"
    echo "  - 'multiversion' gives runtime detection (portable + fast)"
fi
echo ""
echo "Filter examples:"
echo "  ./scripts/bench_compiler_flags.sh '1024b'    # Only 1024-bit benchmarks"
echo "  ./scripts/bench_compiler_flags.sh 'dispatch' # Only dispatch group"

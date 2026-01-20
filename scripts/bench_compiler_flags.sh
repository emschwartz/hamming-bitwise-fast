#!/bin/bash
# Compare benchmark results with different compiler flags
# This script demonstrates the impact of compiler optimization flags
# on the dispatch benchmarks across different vector sizes.
#
# Usage:
#   ./scripts/bench_compiler_flags.sh           # Run all sizes
#   ./scripts/bench_compiler_flags.sh 1024bit   # Run only 1024-bit
#   ./scripts/bench_compiler_flags.sh 512bit    # Run only 512-bit
#
# To run benchmarks manually filtered by size:
#   cargo bench -- "1024bit"                    # All 1024-bit benchmarks
#   cargo bench -- "dispatch.*1024bit"          # Only dispatch group, 1024-bit
#   cargo bench -- "head_to_head.*512bit"       # Only head_to_head, 512-bit

set -e

SIZE_FILTER="${1:-}"  # Optional size filter (e.g., "1024bit")
BENCH_FILTER="dispatch"
if [ -n "$SIZE_FILTER" ]; then
    BENCH_FILTER="dispatch.*$SIZE_FILTER"
    echo "Filtering to size: $SIZE_FILTER"
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

    echo "=== $name ==="
    echo ""

    if [ -n "$rustflags" ]; then
        if [ -n "$features" ]; then
            RUSTFLAGS="$rustflags" cargo bench --bench single_pair --features "$features" -- "$BENCH_FILTER" 2>/dev/null || echo "  [SKIPPED - CPU doesn't support required features]"
        else
            RUSTFLAGS="$rustflags" cargo bench --bench single_pair -- "$BENCH_FILTER" 2>/dev/null || echo "  [SKIPPED - CPU doesn't support required features]"
        fi
    else
        if [ -n "$features" ]; then
            cargo bench --bench single_pair --features "$features" -- "$BENCH_FILTER" 2>/dev/null || echo "  [SKIPPED - feature not available]"
        else
            cargo bench --bench single_pair -- "$BENCH_FILTER" 2>/dev/null
        fi
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

    echo "--- With multiversion feature ---"
    run_bench "2a. multiversion" "" "multiversion"
    run_bench "2b. multiversion + native" "-C target-cpu=native" "multiversion"

    echo "--- With pulp feature ---"
    run_bench "3a. pulp" "" "pulp"
    run_bench "3b. pulp + native" "-C target-cpu=native" "pulp"

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
echo "Each configuration tested with both default and native CPU targeting."
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
echo "Filter benchmarks by size:"
echo "  cargo bench -- '1024bit'           # All groups, 1024-bit only"
echo "  cargo bench -- 'dispatch.*512bit'  # Dispatch group, 512-bit only"

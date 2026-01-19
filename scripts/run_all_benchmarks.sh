#!/bin/bash
#
# Run all benchmark configurations to compare optimization strategies
#
# This script runs benchmarks in different configurations to answer:
# 1. Does the compiler auto-vectorize optimally without hints?
# 2. How much does native target compilation help?
# 3. How much overhead does multiversion dispatch add?
# 4. How do batching and allocation strategies compare?
#
# Results are saved to target/criterion/ with HTML reports

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=============================================="
echo "Hamming Distance Benchmark Suite"
echo "=============================================="
echo ""

# Create results directory
RESULTS_DIR="$PROJECT_DIR/benchmark_results"
mkdir -p "$RESULTS_DIR"

# Get system info
echo "System Information:"
echo "  CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo "  Cores: $(nproc)"

# Check for AVX-512 support
if grep -q avx512 /proc/cpuinfo 2>/dev/null; then
    echo "  AVX-512: Supported"
    AVX512_SUPPORTED=true
else
    echo "  AVX-512: Not supported"
    AVX512_SUPPORTED=false
fi

# Check for AVX2 support
if grep -q avx2 /proc/cpuinfo 2>/dev/null; then
    echo "  AVX2: Supported"
else
    echo "  AVX2: Not supported"
fi

echo ""
echo "=============================================="
echo "Phase 1: Default Compilation (Auto-vectorization)"
echo "=============================================="
echo "This tests whether the compiler auto-vectorizes without explicit target features"
echo ""

cargo bench --bench single_pair 2>&1 | tee "$RESULTS_DIR/single_pair_default.log"

echo ""
echo "=============================================="
echo "Phase 2: Native Target Compilation"
echo "=============================================="
echo "Compiling with -C target-cpu=native to enable all CPU features"
echo ""

RUSTFLAGS="-C target-cpu=native" cargo bench --bench single_pair 2>&1 | tee "$RESULTS_DIR/single_pair_native.log"

echo ""
echo "=============================================="
echo "Phase 3: Multiversion Feature Enabled"
echo "=============================================="
echo "Testing runtime CPU dispatch via multiversion crate"
echo ""

cargo bench --bench single_pair --features multiversion 2>&1 | tee "$RESULTS_DIR/single_pair_multiversion.log"

echo ""
echo "=============================================="
echo "Phase 4: Batch Operations"
echo "=============================================="
echo ""

# Default compilation
cargo bench --bench batch 2>&1 | tee "$RESULTS_DIR/batch_default.log"

# With multiversion
cargo bench --bench batch --features multiversion 2>&1 | tee "$RESULTS_DIR/batch_multiversion.log"

echo ""
echo "=============================================="
echo "Phase 5: Memory & Allocation Strategies"
echo "=============================================="
echo ""

cargo bench --bench memory_allocation 2>&1 | tee "$RESULTS_DIR/memory_allocation.log"

echo ""
echo "=============================================="
echo "Phase 6: Combined Optimization Comparison"
echo "=============================================="
echo ""

# Default
cargo bench --bench combined 2>&1 | tee "$RESULTS_DIR/combined_default.log"

# With multiversion
cargo bench --bench combined --features multiversion 2>&1 | tee "$RESULTS_DIR/combined_multiversion.log"

# Native + multiversion (best possible)
RUSTFLAGS="-C target-cpu=native" cargo bench --bench combined --features multiversion 2>&1 | tee "$RESULTS_DIR/combined_native_multiversion.log"

echo ""
echo "=============================================="
echo "Benchmarks Complete!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - HTML reports: target/criterion/report/index.html"
echo "  - Log files: $RESULTS_DIR/"
echo ""
echo "Key comparisons to make:"
echo "  1. single_pair_default.log vs single_pair_native.log"
echo "     -> Shows if compiler auto-vectorizes to use AVX-512 VPOPCNTDQ"
echo ""
echo "  2. single_pair_native.log vs single_pair_multiversion.log"
echo "     -> Shows multiversion dispatch overhead"
echo ""
echo "  3. combined_default.log vs combined_native_multiversion.log"
echo "     -> Shows total optimization impact"

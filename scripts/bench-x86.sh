#!/usr/bin/env bash
#
# Run the full x86 benchmark matrix used to source README/CHANGELOG claims
# for hamming-bitwise-fast.
#
# Matrix: 4 build configs × 3 bench targets, plus an idle-CPU sanity check.
#
#   C1: default features (multiversion_x86 ON, lto = true)
#   C2: target-cpu=native (multiversion_x86 ON, lto = true, native codegen)
#   C3: lto = false       (multiversion_x86 ON)
#   C4: SSE2 baseline     (multiversion_x86 OFF, lto = true)
#
#   bench targets: competitors, batch_input_type (includes gather_demo), dispatch
#
# Each (config, bench) pair writes:
#   results/<config>/<bench>.log     — full criterion text output
#   target/criterion/<baseline name> — raw measurement data (per baseline)
#
# Why one baseline per config: criterion's `change:` lines compare against the
# previously-run config, so swapping baselines lets us see paired deltas in
# the log (e.g. "lto=false regressed by 600%") on the same hardware.
#
# Estimated wall time: ~25-40 minutes total on a Zen 4/5 server. Each
# `cargo criterion` run does 100 samples × ~5s warmup+collect per benchmark.
#
# Usage:
#   scripts/bench-x86.sh                 # foreground
#   nohup scripts/bench-x86.sh > run.log 2>&1 &
#                                        # detach so it survives SSH drop
#
# Prerequisites on the remote host:
#   - rustup with a recent stable toolchain
#   - cargo install cargo-criterion
#   - this repo checked out, working dir = repo root
#   - quiet machine: no other tenants, governor=performance, turbo decided

set -euo pipefail

# ----------------------------------------------------------------------------
# Sanity
# ----------------------------------------------------------------------------
if [[ ! -f Cargo.toml ]] || ! grep -q '^name = "hamming-bitwise-fast"' Cargo.toml; then
  echo "error: run from the hamming-bitwise-fast repo root" >&2
  exit 1
fi

ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) ;;
  *)
    echo "error: this script is x86-only (detected: $ARCH)" >&2
    echo "       multiversion + lto + native sweep is meaningless on ARM" >&2
    exit 1
    ;;
esac

if ! command -v cargo-criterion >/dev/null 2>&1; then
  echo "error: cargo-criterion not installed. run: cargo install cargo-criterion" >&2
  exit 1
fi

mkdir -p results/c1-default results/c2-native results/c3-no-lto results/c4-sse2

# ----------------------------------------------------------------------------
# Capture environment for the report
# ----------------------------------------------------------------------------
{
  echo "=== host ==="
  uname -a
  echo
  echo "=== cpu ==="
  if [[ -r /proc/cpuinfo ]]; then
    grep -m1 'model name' /proc/cpuinfo || true
    echo
    echo "flags (avx512/popcnt):"
    grep -m1 -o 'avx512[a-z0-9]*\|vpopcntdq\|popcnt' /proc/cpuinfo | sort -u || true
  fi
  echo
  echo "=== governor ==="
  if command -v cpupower >/dev/null 2>&1; then
    cpupower frequency-info 2>&1 | head -20 || true
  fi
  echo
  echo "=== toolchain ==="
  rustc -V
  cargo -V
  cargo-criterion --version || true
} > results/host-info.txt 2>&1

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
# A single (config, bench) pass:
#   $1 = config dir (e.g. c1-default)
#   $2 = bench target name
#   remaining args appended to cargo invocation (features, profile env vars are
#   already in scope when this is called)
#
# Note: cargo-criterion does NOT support --save-baseline (that's a `cargo bench`
# flag for the criterion crate's built-in runner). Instead criterion auto-rotates
# `new` -> `base` between consecutive runs, so the paired `change:` blocks in
# the log already give us "CONFIG 2 vs CONFIG 1" deltas without naming baselines.
# To preserve raw data per-config, the script tar+rsyncs target/criterion/ after
# each config completes.
run_bench() {
  local config="$1"; shift
  local bench="$1";  shift

  echo "==> $config / $bench"
  cargo criterion --bench "$bench" "$@" \
    > "results/$config/$bench.log" 2>&1
}

# Snapshot raw criterion data for a config so later configs don't overwrite it.
snapshot_criterion() {
  local config="$1"
  if [[ -d target/criterion ]]; then
    tar -czf "results/$config/criterion-data.tar.gz" -C target criterion 2>/dev/null || true
  fi
}

# ----------------------------------------------------------------------------
# CONFIG 1: default features, lto=true (the baseline everyone is compared to)
# ----------------------------------------------------------------------------
echo "=== CONFIG 1: default features, lto=true ==="
unset RUSTFLAGS
unset CARGO_PROFILE_BENCH_LTO
run_bench c1-default competitors
run_bench c1-default batch_input_type
run_bench c1-default dispatch
snapshot_criterion c1-default

# ----------------------------------------------------------------------------
# CONFIG 2: target-cpu=native, lto=true
# ----------------------------------------------------------------------------
echo "=== CONFIG 2: target-cpu=native, lto=true ==="
export RUSTFLAGS="-C target-cpu=native"
unset CARGO_PROFILE_BENCH_LTO
run_bench c2-native competitors
run_bench c2-native batch_input_type
run_bench c2-native dispatch
snapshot_criterion c2-native

# ----------------------------------------------------------------------------
# CONFIG 3: lto=false (paired against C1 to isolate the LTO effect)
# ----------------------------------------------------------------------------
echo "=== CONFIG 3: lto=false ==="
unset RUSTFLAGS
export CARGO_PROFILE_BENCH_LTO=false
run_bench c3-no-lto competitors
run_bench c3-no-lto batch_input_type
run_bench c3-no-lto dispatch
snapshot_criterion c3-no-lto

# ----------------------------------------------------------------------------
# CONFIG 4: no-default-features (multiversion OFF → SSE2 baseline)
#   This isolates "what does the multiversion feature actually buy us."
#   `dispatch` is meaningless here because both branches collapse to SSE2,
#   so we skip it.
# ----------------------------------------------------------------------------
echo "=== CONFIG 4: no-default-features (SSE2 baseline) ==="
unset RUSTFLAGS
unset CARGO_PROFILE_BENCH_LTO
run_bench c4-sse2 competitors       --no-default-features
run_bench c4-sse2 batch_input_type  --no-default-features
snapshot_criterion c4-sse2

echo
echo "=== done ==="
echo "Per-config logs:        results/c{1,2,3,4}-*/[bench].log"
echo "Raw criterion data:     target/criterion/  (one subdir per baseline)"
echo "Host info:              results/host-info.txt"
echo
echo "Pull back with:"
echo "  rsync -av REMOTE:hamming-bitwise-fast/results/ ./results-x86/"

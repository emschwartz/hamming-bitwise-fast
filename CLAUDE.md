# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rust crate (`hamming-bitwise-fast`) providing fast bitwise Hamming distance computation for byte arrays/slices. The implementation uses auto-vectorization techniques that enable SIMD optimizations (AVX-512 VPOPCNTDQ on x86, NEON on ARM) without explicit intrinsics.

## Build Commands

```sh
# Build
cargo build
cargo build --release

# Build with x86 multiversion support (runtime CPU dispatch, enabled by default)
cargo build --features multiversion_x86

# Build with native CPU optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Run tests
cargo test

# Run a specific test
cargo test slice_distance_all_bits_different

# Run benchmarks (uses criterion via cargo-criterion for better output and HTML reports)
# Install: cargo install cargo-criterion
cargo criterion

# Run a specific benchmark file
cargo criterion --bench competitors

# Filter benchmarks by name pattern
cargo criterion -- single
```

## Development Guidelines

**Always verify both assembly AND benchmarks after changes.** This crate's performance comes entirely from the compiler generating optimal SIMD instructions. Seemingly minor code changes (e.g., loop structure, iterator patterns, type annotations) can cause the compiler to emit dramatically slower code.

For any performance-related change:
1. **Inspect assembly** on x86 (with AVX-512) to verify expected instructions are generated. See [Inspecting Generated Assembly](#inspecting-generated-assembly).
2. **Run benchmarks on both ARM and x86** to confirm performance hypotheses. Assembly that "looks right" can still be slow, and vice versa.
3. **Understand why** the results are what they are. If benchmarks don't match expectations, investigate until you understand the cause—don't just accept surprising results.

Development is typically done on ARM Mac, but x86 benchmarks require a remote server (the project uses Linode and fly.io).

## Architecture

### Public API

Organized into two modules (`src/array.rs` and `src/slice.rs`):

**`array` module** — fixed-size `[u8; N]` (recommended when size is known):
1. **`array::distance(&[u8; N], &[u8; N]) -> u32`** - Single comparison
2. **`array::batch(&[u8; N], &[[u8; N]], &mut [u32])`** - One-to-many (fastest for bulk)

**`slice` module** — variable-length `&[u8]`:
3. **`slice::distance(&[u8], &[u8]) -> u32`** - Single comparison
4. **`slice::batch(&[u8], &[&[u8]], &mut [u32])`** - One-to-many

### Code Structure

- **`src/lib.rs`** — Two `cfg`-gated versions of `distance_impl()`: x86 uses u64 chunks via `chunks_exact(8)`, non-x86 uses simple byte iteration. Also re-exports `array`/`slice` modules and provides a `hamming_bitwise_fast()` convenience alias.
- **`src/array.rs`** — `distance()` and `batch()` for `[u8; N]`. Contains the `opaque_ptr` asm barrier for gather avoidance (see below). Each function has a `#[cfg_attr(... multiversion::multiversion(...))]` attribute for runtime CPU dispatch on x86.
- **`src/slice.rs`** — `distance()` and `batch()` for `&[u8]`. Same multiversion attribute pattern. No gather barrier needed (pointer-indirect layout).
- **`src/tests.rs`** — Parameterized tests using `test_case` crate.

### Platform Dispatch Strategy

Each public function uses `#[cfg_attr]` to conditionally apply `#[multiversion::multiversion]`:

- **x86/x86_64 with `multiversion_x86` feature** (default): The `multiversion` attribute generates multiple copies targeting AVX-512, AVX2, and SSE4.2, with runtime CPU dispatch via CPUID.
- **x86/x86_64 without feature**: Falls through to `distance_impl` which uses u64-chunked processing with `chunks_exact(8)` for auto-vectorization.
- **ARM/other platforms**: `distance_impl` uses simple byte iteration that auto-vectorizes well with NEON.

### Core Algorithm

The x86 `distance_impl` processes bytes as u64 chunks:
```rust
a.chunks_exact(8).zip(b.chunks_exact(8))
    .map(|(a_chunk, b_chunk)| {
        let a_val = u64::from_ne_bytes(a_chunk.try_into().unwrap());
        let b_val = u64::from_ne_bytes(b_chunk.try_into().unwrap());
        (a_val ^ b_val).count_ones()
    })
    .sum()
```

This pattern enables the compiler to use VPOPCNTDQ on AVX-512 CPUs.

**LTO requirement:** Auto-vectorization depends on the compiler seeing the full loop body. Without LTO, cross-crate MIR inlining doesn't give LLVM enough visibility — it emits scalar POPCNT instead of VPOPCNTDQ. The `[profile.bench] lto = true` in Cargo.toml ensures benchmarks get full optimization. Users should enable `lto = true` in their release profile for best single-call performance.

### Gather Avoidance in Batch Functions

**PERFORMANCE INVARIANT:** `array::batch` uses an `asm!` barrier (`opaque_ptr` in `src/array.rs`) on target references to prevent LLVM from generating slow AVX-512 VPGATHERQQ gather instructions across iterations of the contiguous `&[[u8; N]]` layout.

The barrier's effect depends on LTO:

- **With LTO + multiversion (the recommended user config):** LLVM inlines fully, sees `N` is a compile-time constant, unrolls the inner loop, and emits no gathers either way. The barrier is a verified no-op — assembly is identical with and without it.
- **Without LTO + multiversion:** Each multiversion specialization is a separate translation unit. LLVM can't see `N`, falls back to outer-loop vectorization, and emits VPGATHERQQ across iterations (~112 such instructions in the benchmark binary, ~4× runtime slowdown). The barrier is load-bearing here.

The barrier is kept unconditionally as defense for users who don't enable LTO and as insurance against future LLVM versions changing the heuristic under LTO. Cost under LTO is zero (verified).

Why not `black_box`? Both prevent the gather, but `black_box` adds a ~5-cycle store-forwarding penalty per iteration. Under LTO + AVX-512 that's ~7× slower than the asm! barrier (gather_demo: black_box 2.85µs vs asm_barrier 410ns at 64B).

After modifying batch functions, verify:
1. Inspect x86 AVX-512 assembly under `CARGO_PROFILE_BENCH_LTO=false` for absence of VPGATHERQQ in the barriered loop.
2. Run `cargo criterion --bench batch_input_type` — `gather_demo/asm_barrier_zero_cost` should match `gather_demo/no_blackbox_slow_gather` under LTO, and beat it ~4× under no-LTO.
3. The `gather_demo` benchmark group provides a direct A/B/C comparison (no barrier, `black_box`, asm barrier).

This does NOT affect `slice::batch` — `&[&[u8]]` is pointer-indirect, so gathers aren't possible regardless.

### Benchmark Suite

Benchmarks in `benches/` use the [criterion](https://crates.io/crates/criterion) framework. Run with `cargo criterion` (requires `cargo install cargo-criterion`) for better terminal output and HTML reports. Configuration is in `criterion.toml`.

Key benchmarks:
- `competitors.rs` - Compare against other crates (simsimd, hamming, triple_accel, hamming_rs)
- `batch_input_type.rs` - Array batch vs slice batch, gather demo
- `threshold.rs` / `threshold_strategy.rs` - Early-exit threshold experiments
- `chunk_strategy.rs` - Chunk strategy experiments
- `dispatch.rs` - Dispatch overhead measurements
- `array_vs_slice.rs` - Array vs slice API comparison
- `l1_cache_effect.rs` - L1 cache sizing effects on benchmarks

Shared test data generation is in `benches/helpers.rs` (includes `l1_batch_size()` to keep benchmark inputs within L1 cache).

### Inspecting Generated Assembly

**IMPORTANT:** This is performance-critical code where small changes can cause 2-5x performance regressions. Always inspect the generated assembly after modifying any of the core functions. Look for:
- Vectorized instructions (VPOPCNTDQ, VPXORD, VMOVDQU64 on AVX-512; POPCNT on older x86; CNT on ARM NEON)
- Absence of slow gather instructions (VPGATHERQQ) in batch functions
- Proper loop unrolling

Since all public functions and `distance_impl` use `#[inline(always)]`, you need special techniques to view the generated assembly:

**Option 1: Use `#[no_mangle]` wrapper functions in an example file**

Create `examples/asm_check.rs` with wrapper functions:
```rust
use hamming_bitwise_fast::array;

#[no_mangle]
pub fn hamming_128(a: &[u8; 128], b: &[u8; 128]) -> u32 {
    array::distance(a, b)
}
```

**Option 2: Temporarily add `#[inline(never)]` to lib.rs**

For quick iteration, temporarily change `#[inline(always)]` to `#[inline(never)]` on `distance_impl`.

**Viewing assembly:**
```sh
# Using objdump (after creating an example file)
RUSTFLAGS="-C target-cpu=native" cargo build --release --example asm_check
objdump -d target/release/examples/asm_check | less

# For x86 with specific features
RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+avx512vpopcntdq" cargo build --release --example asm_check
```

**Cross-compiling for x86 (from ARM Mac):**

To inspect x86 assembly when developing on ARM, either:
- Use a remote x86 server (Linode, fly.io, etc.)
- Use `cross` for local cross-compilation:
  ```sh
  cargo install cross
  cross build --release --target x86_64-unknown-linux-gnu --example asm_check
  ```

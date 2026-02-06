# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rust crate (`hamming-bitwise-fast`) providing fast bitwise Hamming distance computation for byte arrays/slices. The implementation uses auto-vectorization techniques that enable SIMD optimizations (AVX-512 VPOPCNTDQ on x86, NEON on ARM) without explicit intrinsics.

## Build Commands

```sh
# Build
cargo build
cargo build --release

# Build with x86 multiversion support (runtime CPU dispatch)
cargo build --features multiversion_x86

# Build with native CPU optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Run tests
cargo test

# Run a specific test
cargo test hamming_bitwise_array_correctness

# Run benchmarks (uses criterion via cargo-criterion for better output and HTML reports)
# Install: cargo install cargo-criterion
cargo criterion

# Run a specific benchmark
cargo criterion --bench multiversion

# Run with multiversion feature
cargo criterion --features multiversion_x86 --bench multiversion

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
3. **`array::threshold(&[u8; N], &[u8; N], u32) -> Option<u32>`** - Early-exit when distance exceeds threshold
4. **`array::batch_threshold(&[u8; N], &[[u8; N]], u32, &mut [u32]) -> u32`** - Batch with early exit

**`slice` module** — variable-length `&[u8]`:
5. **`slice::distance(&[u8], &[u8]) -> u32`** - Single comparison
6. **`slice::batch(&[u8], &[&[u8]], &mut [u32])`** - One-to-many
7. **`slice::threshold(&[u8], &[u8], u32) -> Option<u32>`** - Early-exit
8. **`slice::batch_threshold(&[u8], &[&[u8]], u32, &mut [u32]) -> u32`** - Batch with early exit

### Platform Dispatch Strategy

The crate uses the `define_hamming_fn!` macro to generate three versions of each function:

- **x86/x86_64 with `multiversion_x86` feature**: Uses `#[multiversion::multiversion]` for runtime CPU dispatch (AVX-512, AVX2, SSE4.2)
- **x86/x86_64 without feature**: Uses u64-chunked processing with `chunks_exact(8)` for auto-vectorization
- **ARM/other platforms**: Simple byte iterator that auto-vectorizes well with NEON

### Core Algorithm

The x86 optimization relies on processing bytes as u64 chunks:
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

**Note:** In `array::batch`, the function calls `array::distance` (the public multiversion function) rather than `array_impl` directly. The multiversion dispatch boundary prevents the compiler from seeing the contiguous `&[[u8; N]]` layout and generating slow VPGATHERQQ gather instructions.

### Benchmark Suite

Benchmarks in `benches/` use the [criterion](https://crates.io/crates/criterion) framework. Run with `cargo criterion` (requires `cargo install cargo-criterion`) for better terminal output and HTML reports. Configuration is in `criterion.toml`.

- `competitors.rs` - Compare against other crates (simsimd, hamming, triple_accel, hamming_rs)
- `batch_input_type.rs` - Array batch vs slice batch, gather demo
- `threshold.rs` - Early-exit threshold benchmarks, streaming top-k simulation
- `chunk_strategy.rs` - Chunk strategy experiments
- `dispatch.rs` - Dispatch overhead measurements
- `batch_vs_loop.rs` - Batch operations vs individual calls
- `u64_storage.rs` - u8 vs u64 storage format

Shared test data generation is in `benches/helpers.rs`.

### Inspecting Generated Assembly

**IMPORTANT:** This is performance-critical code where small changes can cause 2-5x performance regressions. Always inspect the generated assembly after modifying any of the core functions. Look for:
- Vectorized instructions (VPOPCNTDQ, VPXORD, VMOVDQU64 on AVX-512; POPCNT on older x86; CNT on ARM NEON)
- Absence of slow gather instructions (VPGATHERQQ) in batch functions
- Proper loop unrolling

Since all public functions use `#[inline(always)]`, you need special techniques to view the generated assembly:

**Option 1: Use `#[no_mangle]` wrapper functions in examples**

Create wrapper functions in `examples/` with `#[no_mangle]` (see `examples/asm_check.rs`):
```rust
#[no_mangle]
pub fn hamming_128(a: &[u8; 128], b: &[u8; 128]) -> u32 {
    array::distance(a, b)
}
```

**Option 2: Temporarily add `#[inline(never)]` to lib.rs**

For quick iteration, temporarily change `#[inline(always)]` to `#[inline(never)]` in the macro.

**Viewing assembly:**
```sh
# Using cargo-asm (install: cargo install cargo-asm)
cargo asm --example asm_inspect --native hamming_

# Using objdump
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

The `Cross.toml` file is already configured for x86_64-unknown-linux-gnu.

### Deprecated API

- **`hamming_bitwise_fast`** - Deprecated alias for `slice::distance` (retained for backwards compatibility)

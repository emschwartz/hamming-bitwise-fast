# Hamming Bitwise Fast

> A fast, zero-dependency\* implementation of bitwise Hamming Distance using
> a method amenable to auto-vectorization.

This started out as a benchmark of various bitwise Hamming distance implementations in Rust.
However, after finding that a simple implementation that is amenable to auto-vectorization
was comparable, if not faster, than other implementations, I decided to publish it as a crate.
A second round of benchmarking uncovered more optimizations and yielded even faster results.

**Note:** This is for comparing bit-vectors, _not_ for comparing strings.

_\* Zero dependencies by default. The optional `multiversion_x86` feature adds the [`multiversion`](https://crates.io/crates/multiversion) crate for runtime SIMD support detection on x86 CPUs. See [SIMD on x86](#simd-on-x86) for details._

## Usage

For variable-length slices:
```rust
use hamming_bitwise_fast::slice;

// Single comparison
let a: Vec<u8> = vec![0xFF; 128];
let b: Vec<u8> = vec![0x00; 128];
let distance: u32 = slice::distance(&a, &b); // 1024

// Batch comparison (one source vs many targets)
// Pre-allocate result vec once and reuse across calls for best performance
let source: Vec<u8> = vec![0x00; 128];
let targets: Vec<&[u8]> = vec![&a, &b];
let mut distances: Vec<u32> = vec![0; 2];
slice::batch(&source, &targets, &mut distances); // [1024, 0]
```

For fixed-size arrays (faster for sizes under 2048 bits / 256 bytes):
```rust
use hamming_bitwise_fast::array;

// Single comparison
let a: [u8; 128] = [0xFF; 128];  // 1024-bit vectors
let b: [u8; 128] = [0x00; 128];
let distance: u32 = array::distance(&a, &b); // 1024

// Batch comparison (one source vs many targets)
// Pre-allocate result vec once and reuse across calls for best performance
let source: [u8; 128] = [0x00; 128];
let targets: Vec<[u8; 128]> = vec![a, b];
let mut distances: Vec<u32> = vec![0; 2];
array::batch(&source, &targets, &mut distances); // [1024, 0]
```

## API

The crate provides two modules:

- `array::distance` / `slice::distance` - single comparisons for fixed-size arrays and slices
- `array::batch` / `slice::batch` - batch comparisons (one source vs many targets)
- `array::threshold` / `slice::threshold` - single comparison with early exit
- `array::batch_threshold` / `slice::batch_threshold` - batch with early exit

## SIMD on x86

> **TL;DR:** On x86, enable the `multiversion_x86` feature for best performance:
> ```sh
> cargo add hamming-bitwise-fast --features multiversion_x86
> ```

### The Problem

Rust targets the baseline [x86-64 microarchitecture level](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels) (v1) by default, which only includes SSE2. This ensures binaries run on any x86-64 CPU made since 2003, but misses major SIMD improvements that can make Hamming distance **4-5x faster**:

| Level | Year | Key Features |
|-------|------|--------------|
| x86-64-v1 | 2003 | SSE2 (baseline) |
| x86-64-v2 | 2008 | SSE4.2, POPCNT |
| x86-64-v3 | 2013 | AVX2, BMI1/2 |
| x86-64-v4 | 2017 | AVX-512 |

**Note:** The `VPOPCNTDQ` instruction (vectorized popcount) that makes Hamming distance extremely fast is a *separate* AVX-512 extension, not part of the base x86-64-v4 level. It's available on Intel Ice Lake (2019+) and AMD Zen 4 (2022+) CPUs. The `multiversion_x86` feature automatically detects and uses it when available.

### The Solution: `multiversion_x86`

The `multiversion_x86` feature uses the [`multiversion`](https://crates.io/crates/multiversion) crate to compile multiple code paths and select the fastest one at runtime via CPUID:

```toml
[dependencies]
hamming-bitwise-fast = { version = "1", features = ["multiversion_x86"] }
```

This gives near-optimal performance on any x86 CPU without risking "illegal instruction" crashes.

### Alternative: Compile-time CPU targeting

If you know your target CPU, you can use `RUSTFLAGS` for slightly better performance (no runtime dispatch overhead):

```sh
# Best performance, but binary only runs on identical CPUs
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Requires AVX2 (2013+ CPUs)
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release

# Requires AVX-512 base (2017+ server CPUs)
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo build --release

# Or explicitly enable the feature on any AVX-512 base:
RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+avx512vpopcntdq" cargo build --release
```

> ⚠️ **Warning:** Binaries built with compile-time CPU targeting will crash with "illegal instruction" if run on a CPU that doesn't support the required features. For portable deployments, use the `multiversion_x86` feature instead.

### Performance comparison (AMD EPYC Zen 4)

| Option | Speed (128 bytes / 1024-bit) |
|--------|------------------------------|
| Default (no multiversion) | ~8ns |
| `multiversion_x86` feature | ~2.5ns |
| `multiversion_x86` + `batch` | ~2.2ns per comparison |

**For Docker/cloud deployments**, `multiversion_x86` is strongly recommended—it automatically uses AVX-512 on modern cloud instances while remaining compatible with older hardware.

## Benchmarks

Comparing against other Hamming distance crates:
[`simsimd`](https://crates.io/crates/simsimd),
[`hamming`](https://crates.io/crates/hamming),
[`triple_accel`](https://crates.io/crates/triple_accel),
[`hamming_rs`](https://crates.io/crates/hamming_rs) (x86 only)

### Single Comparison

#### ARM (Apple M2 Max)

| Function | 64 bytes | 128 bytes | 256 bytes |
|----------|----------|-----------|-----------|
| **array::distance** | **1.2ns** | **2.2ns** | **4.3ns** |
| **slice::distance** | **1.8ns** | **2.7ns** | **5.0ns** |
| simsimd | 4.7ns | 6.5ns | 10.4ns |
| triple_accel | 7.4ns | 11.9ns | 21.2ns |
| hamming | 7.5ns | 11.9ns | 18.6ns |

#### x86 (with `multiversion_x86` feature)

| Function | 64 bytes | 128 bytes | 256 bytes |
|----------|----------|-----------|-----------|
| **array::distance** | **2.4ns** | **2.7ns** | **4.4ns** |
| **slice::distance** | **2.4ns** | **3.1ns** | **4.0ns** |
| triple_accel | 3.5ns | 4.0ns | 5.5ns |
| simsimd | 3.8ns | 4.5ns | 5.9ns |
| hamming_rs | 15ns | 24ns | 41ns |
| hamming | 47ns | 94ns | 29ns |

### Batch Comparison (1000 comparisons)

The batch functions are faster than calling single functions in a loop for one-to-many comparisons.

#### ARM (Apple M2 Max)

| Function | 64 bytes | 128 bytes | 256 bytes |
|----------|----------|-----------|-----------|
| **array::batch** | **1.3µs** | **2.2µs** | **4.5µs** |
| array::distance (loop) | 1.4µs | 2.5µs | 4.7µs |
| **slice::batch** | **1.8µs** | **2.7µs** | **4.8µs** |
| slice::distance (loop) | 2.2µs | 3.1µs | 5.6µs |
| simsimd | 4.8µs | 6.6µs | 10.5µs |
| triple_accel | 7.7µs | 12.0µs | 21.1µs |

#### x86 (with `multiversion_x86` feature)

| Function | 64 bytes | 128 bytes | 256 bytes |
|----------|----------|-----------|-----------|
| **slice::batch** | **1.4µs** | **2.2µs** | **3.7µs** |
| slice::distance (loop) | 2.7µs | 3.5µs | 5.2µs |
| **array::batch** | **2.4µs** | **4.7µs** | **9.4µs** |
| array::distance (loop) | 3.5µs | 4.6µs | 11.7µs |
| simsimd | 4.2µs | 4.8µs | 5.7µs |
| triple_accel | 4.2µs | 4.5µs | 5.5µs |
| hamming_rs | 15µs | 23µs | 41µs |
| hamming | 47µs | 94µs | 29µs |

### Running benchmarks

```sh
# Run all benchmarks
cargo bench

# Run competitor comparison
cargo bench --bench competitors

# With multiversion (x86 only)
cargo bench --features multiversion_x86 --bench competitors
```

## License

This project is licensed under either of the following licenses, at your option:

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

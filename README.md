# Hamming Bitwise Fast

> Fast bitwise Hamming distance for byte arrays/slices, using auto-vectorization
> with runtime SIMD detection on x86.

**Note:** This is for comparing bit-vectors, _not_ for comparing strings.

On x86, enable LTO for best performance — see [Performance on x86](#performance-on-x86).

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

For fixed-size arrays (recommended when size is known at compile time):

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

- `array::distance` / `slice::distance` — single comparison
- `array::batch` / `slice::batch` — batch comparison (one source vs many targets)

## Benchmarks

Compared against other Hamming distance crates:
[`simsimd`](https://crates.io/crates/simsimd),
[`hamming`](https://crates.io/crates/hamming),
[`triple_accel`](https://crates.io/crates/triple_accel),
[`hamming_rs`](https://crates.io/crates/hamming_rs) (x86 only)

### Single Comparison

#### ARM (Apple M2 Max)

| Function            | 64 bytes  | 128 bytes | 256 bytes |
| ------------------- | --------- | --------- | --------- |
| **array::distance** | **1.3ns** | **2.5ns** | **4.3ns** |
| **slice::distance** | **1.3ns** | **2.5ns** | **4.3ns** |
| v1 (baseline)       | 1.6ns     | 3.3ns     | 6.2ns     |
| simsimd             | 4.6ns     | 6.2ns     | 10.3ns    |
| triple_accel        | 4.9ns     | 9.7ns     | 19.5ns    |
| hamming             | 8.1ns     | 12.3ns    | 19.7ns    |

#### x86 (AMD EPYC 9845 Zen 5, with `lto = true`)

| Function            | 64 bytes   | 128 bytes  | 256 bytes  |
| ------------------- | ---------- | ---------- | ---------- |
| **array::distance** | **1.8ns**  | **2.5ns**  | **4.0ns**  |
| **slice::distance** | **2.7ns**  | **3.5ns**  | **5.0ns**  |
| triple_accel        | 2.7ns      | 3.3ns      | 5.0ns      |
| simsimd             | 3.2ns      | 4.1ns      | 6.5ns      |
| v1 (baseline)       | 4.3ns      | 9.1ns      | 18.1ns     |
| hamming_rs          | 5.5ns      | 20.0ns     | 16.3ns     |
| hamming             | 48ns       | 96ns       | 28ns       |

### Batch Comparison (1000 comparisons)

The batch functions are faster for one-to-many comparisons.

#### ARM (Apple M2 Max)

| Function         | 64 bytes  | 128 bytes | 256 bytes |
| ---------------- | --------- | --------- | --------- |
| **array::batch** | **1.3µs** | **2.3µs** | **4.6µs** |
| **slice::batch** | **1.9µs** | **3.1µs** | **5.6µs** |
| simsimd          | 4.6µs     | 6.6µs     | 10.6µs    |
| triple_accel     | 7.5µs     | 12.2µs    | 21.9µs    |
| hamming          | 8.4µs     | 12.5µs    | 19.9µs    |
| v1 (baseline)    | 5.2µs     | 8.9µs     | 12.6µs    |

#### x86 (AMD EPYC 9845 Zen 5, with `lto = true`)

| Function         | 64 bytes  | 128 bytes | 256 bytes |
| ---------------- | --------- | --------- | --------- |
| **array::batch** | **454ns** | **870ns** | **1.6µs** |
| **slice::batch** | **3.9µs** | **6.2µs** | **5.8µs** |
| triple_accel     | 4.0µs     | 4.6µs     | 6.2µs     |
| simsimd          | 4.8µs     | 5.6µs     | 6.4µs     |
| v1 (baseline)    | 9.7µs     | 10.8µs    | 20.5µs    |
| hamming_rs       | 16.7µs    | 25.3µs    | 43.1µs    |
| hamming          | 49.2µs    | 96.6µs    | 31.5µs    |

### Running benchmarks

```sh
# Run competitor comparison
cargo bench --bench competitors
```

## Performance on x86

### Enable LTO

**This is the single most impactful optimization.** This crate relies on
auto-vectorization — the compiler widens u64 operations into SIMD instructions
(e.g., AVX-512 VPOPCNTDQ). Without LTO, the compiler can't see through the crate
boundary to vectorize effectively.

```toml
[profile.release]
lto = true
```

Use full LTO (`true`), not thin (`"thin"`) — thin LTO doesn't give LLVM enough
cross-module visibility for auto-vectorization.

#### Impact on single-call performance (128 bytes, AMD EPYC 9845 Zen 5)

| Configuration              | `array::distance` | `slice::distance` |
| -------------------------- | -----------------:| -----------------:|
| Default (no LTO)           | 9.7ns             | 6.6ns             |
| **Default + `lto = true`** | **2.5ns (3.9x)**  | **3.5ns (1.9x)**  |

**Note on `slice::batch`:** LTO improves single-call and `array::batch` performance
but currently _hurts_ `slice::batch` throughput on x86 (~2-3x slower). If `slice::batch`
is your hot path, benchmark with and without LTO for your workload. `array::batch` is
unaffected and is the fastest option when the size is known at compile time.

### Runtime SIMD detection

Runtime SIMD detection is enabled by default via the [`multiversion`](https://crates.io/crates/multiversion) crate. Rust targets baseline x86-64 (SSE2 only) by default; the `multiversion_x86` feature compiles multiple code paths and selects the fastest at runtime via CPUID — including AVX-512 VPOPCNTDQ on Intel Ice Lake (2019+) and AMD Zen 4+ (2022+) CPUs.

```sh
cargo add hamming-bitwise-fast
```

To disable (e.g., for zero dependencies or when using `-C target-cpu=native`):

```toml
[dependencies]
hamming-bitwise-fast = { version = "1", default-features = false }
```

### Compile-time CPU targeting

If you know your target CPU, combine `lto = true` with `RUSTFLAGS` for maximum performance:

```sh
# Best performance, but binary only runs on identical CPUs
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Requires AVX-512 (2017+ server CPUs)
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo build --release
```

> Binaries built with compile-time CPU targeting will crash with "illegal instruction" if run on a CPU that doesn't support the required features. For portable deployments, use the default `multiversion_x86` feature instead.

## License

This project is licensed under either of the following licenses, at your option:

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

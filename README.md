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
| **array::distance** | **1.3ns** | **2.6ns** | **4.4ns** |
| **slice::distance** | **1.3ns** | **2.5ns** | **4.4ns** |
| simsimd             | 5.2ns     | 7.0ns     | 10.9ns    |
| triple_accel        | 7.6ns     | 12.5ns    | 21.7ns    |
| hamming             | 8.2ns     | 12.4ns    | 19.1ns    |

#### x86 (AMD EPYC Zen 4, with `lto = true`)

| Function            | 64 bytes   | 128 bytes  | 256 bytes  |
| ------------------- | ---------- | ---------- | ---------- |
| **array::distance** | **1.8ns**  | **2.5ns**  | **3.9ns**  |
| **slice::distance** | **2.7ns**  | **3.5ns**  | **5.0ns**  |
| triple_accel        | 2.8ns      | 3.3ns      | 5.0ns      |
| simsimd             | 3.2ns      | 4.2ns      | 6.6ns      |
| hamming_rs          | 5.5ns      | 20.1ns     | 16.4ns     |
| hamming             | 48ns       | 96ns       | 28ns       |

### Batch Comparison (1000 comparisons)

The batch functions are faster for one-to-many comparisons.

#### ARM (Apple M2 Max)

| Function         | 64 bytes  | 128 bytes | 256 bytes |
| ---------------- | --------- | --------- | --------- |
| **array::batch** | **1.3µs** | **2.2µs** | **4.7µs** |
| **slice::batch** | **1.9µs** | **3.0µs** | **5.3µs** |
| simsimd          | 5.1µs     | 7.3µs     | 11.1µs    |
| triple_accel     | 7.7µs     | 12.7µs    | 22.0µs    |
| hamming          | 8.3µs     | 12.8µs    | 19.6µs    |

#### x86 (AMD EPYC Zen 4)

| Function         | 64 bytes  | 128 bytes | 256 bytes |
| ---------------- | --------- | --------- | --------- |
| **array::batch** | **419ns** | **887ns** | **1.7µs** |
| **slice::batch** | **1.4µs** | **2.4µs** | **3.3µs** |
| triple_accel     | 4.5µs     | 5.0µs     | 6.4µs     |
| simsimd          | 4.8µs     | 5.8µs     | 6.6µs     |
| hamming_rs       | 17.2µs    | 26.0µs    | 43.8µs    |
| hamming          | 49.3µs    | 96.7µs    | 31.4µs    |

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

#### Impact on single-call performance (128 bytes, AMD EPYC Zen 4)

| Configuration              | `array::distance` | `slice::distance` |
| -------------------------- | -----------------:| -----------------:|
| Default (no LTO)           | 7.2ns             | 3.2ns             |
| **Default + `lto = true`** | **2.5ns (2.9x)**  | **3.5ns**         |

Batch functions already amortize per-call overhead, so LTO helps less there.

### Runtime SIMD detection

Runtime SIMD detection is enabled by default via the [`multiversion`](https://crates.io/crates/multiversion) crate. Rust targets baseline x86-64 (SSE2 only) by default; the `multiversion_x86` feature compiles multiple code paths and selects the fastest at runtime via CPUID — including AVX-512 VPOPCNTDQ on Intel Ice Lake (2019+) and AMD Zen 4 (2022+) CPUs.

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

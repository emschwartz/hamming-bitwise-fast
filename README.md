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
[`hamming_rs`](https://crates.io/crates/hamming_rs) (Linux/Windows only; AVX2 on x86 for ≥1024 bytes, v0.2.25+ uses this crate's v1.0 algorithm as fallback)

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
| **array::distance** | **1.8ns**  | **2.0ns**  | **2.8ns**  |
| **slice::distance** | **2.7ns**  | **2.8ns**  | **3.6ns**  |
| triple_accel        | 2.7ns      | 3.3ns      | 4.7ns      |
| simsimd             | 3.2ns      | 3.6ns      | 4.7ns      |
| v1 (baseline)       | 4.3ns      | 9.2ns      | 18.2ns     |
| hamming_rs          | 4.3ns      | 8.4ns      | 18.4ns     |
| hamming             | 48ns       | 96ns       | 28ns¹      |

¹ The `hamming` crate's `distance_fast` hits a vectorized fast path at 256 bytes
when its input happens to be sufficiently aligned. The 64 B and 128 B numbers
reflect its slow path. Treat the 256 B cell as a best-case outlier.

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
| **array::batch** | **414ns** | **814ns** | **1.7µs** |
| triple_accel     | 4.0µs     | 4.7µs     | 6.5µs     |
| simsimd          | 5.1µs     | 5.7µs     | 7.5µs     |
| **slice::batch** | **5.2µs** | **6.3µs** | **7.7µs** |
| hamming_rs       | 5.9µs     | 10.3µs    | 19.1µs    |
| v1 (baseline)    | 9.7µs     | 10.8µs    | 20.6µs    |
| hamming          | 49µs      | 97µs      | 32µs¹     |

¹ See the `hamming` footnote in the single-comparison table above.

`array::batch` is the fastest option when sizes are known at compile time. For
runtime-sized inputs, `slice::batch` is competitive with `simsimd` and faster
than every other competitor, but is 5–13× slower than `array::batch` (see the
[LTO note](#impact-on-batch-performance) for why).

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
| Default (no LTO)           | 6.1ns             | 3.4ns             |
| **Default + `lto = true`** | **2.0ns (3.1x)**  | **2.8ns (1.2x)**  |

#### Impact on batch performance

LTO is _not_ universally a win for batch workloads. It modestly improves
`array::batch` (a few hundred ns at most), but currently _hurts_ `slice::batch`
by 2–3× because LLVM picks a less favorable vectorization shape when it can
see across the crate boundary.

128-byte vectors, 1000-comparison batch, AMD EPYC 9845 Zen 5:

| Configuration              | `array::batch`   | `slice::batch`     |
| -------------------------- | ----------------:| ------------------:|
| Default (no LTO)           | 1.10µs           | **2.3µs**          |
| **Default + `lto = true`** | **814ns (1.4x)** | 6.3µs (**2.7x slower**) |

If `slice::batch` is your hot path and you can't switch to `array::batch`,
benchmark with and without LTO for your specific workload. When sizes are
known at compile time, `array::batch` is always the right answer.

### Runtime SIMD detection

Runtime SIMD detection is enabled by default via the [`multiversion`](https://crates.io/crates/multiversion) crate. Rust targets baseline x86-64 (SSE2 only) by default; the `multiversion_x86` feature compiles multiple code paths and selects the fastest at runtime via CPUID — including AVX-512 VPOPCNTDQ on Intel Ice Lake (2019+) and AMD Zen 4+ (2022+) CPUs.

```sh
cargo add hamming-bitwise-fast
```

#### What the feature buys you

128-byte vectors, `lto = true`, AMD EPYC 9845 Zen 5:

| Configuration                  | `array::distance` | `array::batch` (1000) |
| ------------------------------ | -----------------:| ---------------------:|
| `default-features = false` (SSE2 only) | 9.2ns     | 8.9µs                 |
| **default (`multiversion_x86`)**       | **2.0ns (4.7x)** | **814ns (11x)**       |

The dispatch overhead itself is negligible: when both paths produce the same
AVX-512 codegen (e.g., compiled with `target-cpu=native`), the multiversion'd
function and a statically-compiled equivalent are within 1% of each other.
The win above comes entirely from CPUID picking VPOPCNTDQ over SSE2.

To disable (e.g., for zero dependencies or when using `-C target-cpu=native`):

```toml
[dependencies]
hamming-bitwise-fast = { version = "1", default-features = false }
```

### Compile-time CPU targeting

If you know your target CPU, combine `lto = true` with `RUSTFLAGS` for the
fastest single-call performance:

```sh
# Best single-call performance, but binary only runs on identical CPUs
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Requires AVX-512 (2017+ server CPUs)
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo build --release
```

> Binaries built with compile-time CPU targeting will crash with "illegal instruction" if run on a CPU that doesn't support the required features. For portable deployments, use the default `multiversion_x86` feature instead.

**Caveat for batch workloads:** `target-cpu=native` makes single-call faster
(~45% on Zen 5) but currently makes the batch APIs _slower_ at larger sizes.
On a 256-byte vector, 1000-comparison batch:

| Configuration                  | `array::batch` | `slice::batch` |
| ------------------------------ | --------------:| --------------:|
| default features + `lto = true` (multiversion) | **1.7µs**  | **7.7µs**  |
| `target-cpu=native` + `lto = true`             | 1.9µs (+7%) | 11.6µs (+50%) |

LLVM picks a different (worse) vectorization shape for batch loops when
compiled with native than when going through the multiversion dispatch
boundary. If batch throughput matters, prefer the default features over
`target-cpu=native`, or benchmark both for your specific workload.

## License

This project is licensed under either of the following licenses, at your option:

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

# Hamming Bitwise Fast

> A fast, zero-dependency implementation of bitwise Hamming Distance using
> a method amenable to auto-vectorization.

This started out as a benchmark of various bitwise Hamming distance implementations in Rust.
However, after finding that a simple implementation that is amenable to auto-vectorization
was comparable, if not faster, than other implementations, I decided to publish it as a crate.

**Note:** This is for comparing bit-vectors, _not_ for comparing strings.

## Usage

For variable-length slices:
```rust
use hamming_bitwise_fast::hamming_bitwise_slice;

assert_eq!(hamming_bitwise_slice(&[0xFF; 128], &[0xFF; 128]), 0);
assert_eq!(hamming_bitwise_slice(&[0xFF; 128], &[0x00; 128]), 1024);
```

For fixed-size arrays (10-100% faster for sizes under 2048 bits / 256 bytes):
```rust
use hamming_bitwise_fast::hamming_bitwise_array;

let a: [u8; 128] = [0xFF; 128];  // 1024-bit embedding
let b: [u8; 128] = [0x00; 128];
assert_eq!(hamming_bitwise_array(&a, &b), 1024);
```

For batch comparisons (one source vs many targets):
```rust
use hamming_bitwise_fast::hamming_bitwise_array_batch;

let source: [u8; 128] = [0; 128];
let targets = vec![[0xFF; 128], [0; 128]];
let mut distances = vec![0u32; 2];
hamming_bitwise_array_batch(&source, &targets, &mut distances);
assert_eq!(distances, vec![1024, 0]);
```

## Performance

| Function | Best for | Why |
|----------|----------|-----|
| `hamming_bitwise_array` | Fixed-size embeddings < 256 bytes | Compile-time size enables loop unrolling |
| `hamming_bitwise_slice` | Variable-length or large (≥256 byte) data | Simpler API; performance matches array at large sizes |
| `hamming_bitwise_array_batch` | One-to-many comparisons | Amortizes function call overhead |

**x86 performance options:**

| Option | Speed (1024-bit) | Trade-off |
|--------|------------------|-----------|
| Default (`cargo build --release`) | ~9ns | Maximum portability, but slow |
| `multiversion_x86` feature | ~4ns | Fast on any x86 CPU (runtime detection) |
| `-C target-cpu=native` | ~2ns | Fastest, but binary only runs on identical CPUs |
| `-C target-cpu=x86-64-v3` | ~3ns | Requires AVX2 (2013+ CPUs) |
| `-C target-cpu=x86-64-v4` | ~2ns | Requires AVX-512 (2017+ CPUs) |

### Why is the default so slow?

Rust targets the baseline [x86-64 microarchitecture level](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels) (v1) by default, which only includes SSE2. This ensures binaries run on any x86-64 CPU made since 2003, but misses major SIMD improvements:

- **x86-64-v2** (2008+): SSE4.2, POPCNT
- **x86-64-v3** (2013+): AVX2, BMI1/2
- **x86-64-v4** (2017+): AVX-512

You can target a specific level with `-C target-cpu=x86-64-v3`, but the binary will crash with an "illegal instruction" error on older CPUs that lack those features.

**For Docker/cloud deployments**, the `multiversion_x86` feature is recommended—it compiles multiple code paths and selects the fastest one at runtime via CPUID, giving near-optimal performance on any x86 CPU without risking illegal instruction errors.

## Benchmarks

This uses [Criterion](https://github.com/bheisler/criterion.rs) to benchmark various Hamming distance implementations:

- The auto-vectorized implementation in this crate
- A naive for-loop based implementation
- A naive iterator based implementation
- [`hamming`](https://crates.io/crates/hamming) ![hamming](https://img.shields.io/crates/d/hamming)
- [`hamming_rs`](https://crates.io/crates/hamming_rs) ![hamming_rs](https://img.shields.io/crates/d/hamming_rs)
- [`simsimd`](https://crates.io/crates/simsimd) ![simsimd](https://img.shields.io/crates/d/simsimd)

### Running the benchmark

```sh
cargo bench
```

Then open the `target/criterion/report/index.html` file in your browser to view the results.

### Results

#### Single Pair Comparison (Linode 2 CPU 4GB, AVX-512)

Comparing `hamming_bitwise_array` and `hamming_bitwise_slice` against competitor crates:

![Single pair benchmark](results/violin-single-linode-avx512.svg)

#### Batch Comparison (64 targets, Linode 2 CPU 4GB, AVX-512)

Comparing batch operations against looping over individual calls:

![Batch benchmark](results/violin-batch-linode-avx512.svg)

#### Historical Results (older benchmarks)

<details>
<summary>2023 MacBook Pro M2 Max</summary>

![Benchmark results](results/line-chart-macbook.svg)
![Benchmark results](results/violin-chart-macbook.svg)

</details>

<details>
<summary>Linode 2 CPU 4GB</summary>

![Benchmark results](results/line-chart-linode.svg)
![Benchmark results](results/violin-chart-linode.svg)

</details>

<details>
<summary>Fly.io 2 CPU 4GB</summary>

![Benchmark results](results/line-chart-fly.svg)
![Benchmark results](results/violin-chart-fly.svg)

</details>

## License

This project is licensed under either of the following licenses, at your option:

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

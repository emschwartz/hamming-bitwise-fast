//! Benchmarks investigating the known-size performance advantage.
//!
//! Three variants answer two key questions:
//!
//! 1. **Does `array::distance` beat `slice::distance` when both have known-size data?**
//!    (Tests the const-generic vs inlined-slice codegen difference.)
//!
//! 2. **How much does known size help?**
//!    (Compares slice with known-size arrays vs slice with dynamic Vecs.)
//!
//! All benchmarks use `iter_batched_ref` to generate fresh random inputs per
//! batch, avoiding `black_box`-on-inputs asymmetry.
//!
//! Run with: cargo criterion --bench array_vs_slice

mod helpers;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use hamming_bitwise_fast::{array, slice};
use helpers::{l1_batch_size, random_bytes, random_bytes_vec};

fn single_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_distance");

    macro_rules! bench_size {
        ($size:expr) => {{
            let bits = format!("{}b", $size * 8);

            // 1. array::distance — const generic, strongest optimization
            group.bench_function(BenchmarkId::new("array", &bits), |b| {
                b.iter_batched_ref(
                    || (random_bytes::<$size>(), random_bytes::<$size>()),
                    |(a, b)| array::distance(a, b),
                    l1_batch_size(2 * $size),
                )
            });

            // 2. slice::distance with arrays — known size erased by .as_slice() conversion
            group.bench_function(BenchmarkId::new("slice (known size)", &bits), |b| {
                b.iter_batched_ref(
                    || (random_bytes::<$size>(), random_bytes::<$size>()),
                    |(a, b)| slice::distance(a.as_slice(), b.as_slice()),
                    l1_batch_size(2 * $size),
                )
            });

            // 3. slice::distance with dynamic Vecs — compiler doesn't know length
            group.bench_function(BenchmarkId::new("slice (dynamic)", &bits), |b| {
                b.iter_batched_ref(
                    || (random_bytes_vec($size), random_bytes_vec($size)),
                    |(a, b)| slice::distance(a.as_slice(), b.as_slice()),
                    l1_batch_size(2 * $size),
                )
            });
        }};
    }

    bench_size!(64);
    bench_size!(128);
    bench_size!(256);

    group.finish();
}

criterion_group!(benches, single_distance);
criterion_main!(benches);

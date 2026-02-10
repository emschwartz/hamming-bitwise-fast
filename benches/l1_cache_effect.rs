//! Benchmarks measuring the L1 cache effect on performance.
//!
//! Two questions:
//! 1. **Benchmark infrastructure**: Does iter_batched_ref with SmallInput (~1000)
//!    inflate single-distance measurements vs L1-fitting batch sizes?
//! 2. **User-facing batch API**: Would splitting array::batch into L1-sized chunks
//!    help when the targets array overflows L1 cache?
//!
//! Run with: cargo criterion --bench l1_cache_effect

mod helpers;

use criterion::{
    criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use hamming_bitwise_fast::array;
use helpers::{l1_batch_size, random_bytes, random_bytes_array};

use std::hint::black_box;

// ============================================================================
// Question 1: Does iter_batched_ref batch size affect single-distance timing?
// ============================================================================

fn single_batch_size_effect(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_batch_size");

    macro_rules! bench_size {
        ($size:expr) => {{
            let bits = format!("{}b", $size * 8);
            group.throughput(Throughput::Elements(1));

            let setup = || (random_bytes::<$size>(), random_bytes::<$size>());

            // SmallInput: criterion picks ~1000 iterations
            // For 128-byte arrays: 2 * 128 * 1000 = 256KB >> L1 cache
            group.bench_function(BenchmarkId::new("SmallInput", &bits), |b| {
                b.iter_batched_ref(
                    setup,
                    |(a, b)| array::distance(a, b),
                    BatchSize::SmallInput,
                )
            });

            // L1-sized: fits in 32KB L1 cache
            // For 128-byte arrays: 2 * 128 * 128 = 32KB
            group.bench_function(BenchmarkId::new("L1_sized", &bits), |b| {
                b.iter_batched_ref(
                    setup,
                    |(a, b)| array::distance(a, b),
                    l1_batch_size(2 * $size),
                )
            });

            // NumIterations(1): baseline, always L1-hot
            group.bench_function(BenchmarkId::new("single_iter", &bits), |b| {
                b.iter_batched_ref(
                    setup,
                    |(a, b)| array::distance(a, b),
                    BatchSize::NumIterations(1),
                )
            });
        }};
    }

    bench_size!(64);
    bench_size!(128);
    bench_size!(256);

    group.finish();
}

// ============================================================================
// Question 2: Does L1-chunking help array::batch with many targets?
// ============================================================================

fn batch_chunking_effect(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_chunking");

    macro_rules! bench_size {
        ($size:expr) => {{
            let label = format!("{}b", $size * 8);

            let source: [u8; $size] = random_bytes();
            let targets: Vec<[u8; $size]> = random_bytes_array(1000);
            let mut out = vec![0u32; 1000];

            group.throughput(Throughput::Elements(1000));

            // All at once: 1000 targets * N bytes = may overflow L1
            group.bench_with_input(
                BenchmarkId::new("all_at_once", &label),
                &$size,
                |bencher, _| {
                    bencher.iter(|| {
                        array::batch(
                            black_box(&source),
                            black_box(&targets),
                            black_box(&mut out),
                        );
                        black_box(out[0])
                    })
                },
            );

            // L1-chunked: process in chunks that fit source + targets in L1
            const L1_BYTES: usize = 32 * 1024;
            let chunk_size = ((L1_BYTES - $size) / $size).max(1);
            group.bench_with_input(
                BenchmarkId::new(format!("l1_chunked({})", chunk_size), &label),
                &$size,
                |bencher, _| {
                    bencher.iter(|| {
                        for (chunk_targets, chunk_out) in black_box(&targets)
                            .chunks(chunk_size)
                            .zip(black_box(&mut out).chunks_mut(chunk_size))
                        {
                            array::batch(
                                black_box(&source),
                                chunk_targets,
                                chunk_out,
                            );
                        }
                        black_box(out[0])
                    })
                },
            );

            // Also try 2x and 4x L1 chunks for comparison
            let chunk_2x = (chunk_size * 2).min(1000);
            group.bench_with_input(
                BenchmarkId::new(format!("2x_l1_chunked({})", chunk_2x), &label),
                &$size,
                |bencher, _| {
                    bencher.iter(|| {
                        for (chunk_targets, chunk_out) in black_box(&targets)
                            .chunks(chunk_2x)
                            .zip(black_box(&mut out).chunks_mut(chunk_2x))
                        {
                            array::batch(
                                black_box(&source),
                                chunk_targets,
                                chunk_out,
                            );
                        }
                        black_box(out[0])
                    })
                },
            );
        }};
    }

    bench_size!(64);
    bench_size!(128);
    bench_size!(256);

    group.finish();
}

criterion_group!(benches, single_batch_size_effect, batch_chunking_effect);
criterion_main!(benches);

//! Benchmark: can we use threshold(u32::MAX) as a drop-in for distance()?
//!
//! Tests whether the compiler optimizes away the dead branch in the threshold
//! implementation when threshold = u32::MAX, making it equivalent in performance
//! to the dedicated distance implementation.
//!
//! Run with: cargo bench --bench simplify

mod helpers;

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use hamming_bitwise_fast::array;
use helpers::{random_bytes, random_bytes_array};

const BATCH: usize = 1000;

fn single_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("single");

    // 1024-bit vectors (128 bytes) — most common embedding size
    {
        let source: [u8; 128] = random_bytes();
        let targets: Vec<[u8; 128]> = random_bytes_array(BATCH);

        group.bench_function("distance_1024b", |bencher| {
            bencher.iter(|| {
                let mut total = 0u32;
                for target in black_box(&targets).iter() {
                    total += array::distance(black_box(&source), target);
                }
                black_box(total)
            })
        });

        group.bench_function("threshold_max_1024b", |bencher| {
            bencher.iter(|| {
                let mut total = 0u32;
                for target in black_box(&targets).iter() {
                    total += array::threshold(black_box(&source), target, u32::MAX).unwrap();
                }
                black_box(total)
            })
        });
    }

    // 2048-bit vectors (256 bytes)
    {
        let source: [u8; 256] = random_bytes();
        let targets: Vec<[u8; 256]> = random_bytes_array(BATCH);

        group.bench_function("distance_2048b", |bencher| {
            bencher.iter(|| {
                let mut total = 0u32;
                for target in black_box(&targets).iter() {
                    total += array::distance(black_box(&source), target);
                }
                black_box(total)
            })
        });

        group.bench_function("threshold_max_2048b", |bencher| {
            bencher.iter(|| {
                let mut total = 0u32;
                for target in black_box(&targets).iter() {
                    total += array::threshold(black_box(&source), target, u32::MAX).unwrap();
                }
                black_box(total)
            })
        });
    }

    // Small vectors: 64-bit (8 bytes) — threshold block size > input
    {
        let source: [u8; 8] = random_bytes();
        let targets: Vec<[u8; 8]> = random_bytes_array(BATCH);

        group.bench_function("distance_64b", |bencher| {
            bencher.iter(|| {
                let mut total = 0u32;
                for target in black_box(&targets).iter() {
                    total += array::distance(black_box(&source), target);
                }
                black_box(total)
            })
        });

        group.bench_function("threshold_max_64b", |bencher| {
            bencher.iter(|| {
                let mut total = 0u32;
                for target in black_box(&targets).iter() {
                    total += array::threshold(black_box(&source), target, u32::MAX).unwrap();
                }
                black_box(total)
            })
        });
    }

    group.finish();
}

fn batch_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch");

    // 1024-bit
    {
        let source: [u8; 128] = random_bytes();
        let targets: Vec<[u8; 128]> = random_bytes_array(BATCH);

        group.bench_function("batch_1024b", |bencher| {
            let mut out = vec![0u32; BATCH];
            bencher.iter(|| {
                array::batch(black_box(&source), black_box(&targets), &mut out);
                black_box(out[0])
            })
        });

        group.bench_function("batch_threshold_max_1024b", |bencher| {
            let mut out = vec![0u32; BATCH];
            bencher.iter(|| {
                array::batch_threshold(
                    black_box(&source),
                    black_box(&targets),
                    u32::MAX,
                    &mut out,
                );
                black_box(out[0])
            })
        });
    }

    // 2048-bit
    {
        let source: [u8; 256] = random_bytes();
        let targets: Vec<[u8; 256]> = random_bytes_array(BATCH);

        group.bench_function("batch_2048b", |bencher| {
            let mut out = vec![0u32; BATCH];
            bencher.iter(|| {
                array::batch(black_box(&source), black_box(&targets), &mut out);
                black_box(out[0])
            })
        });

        group.bench_function("batch_threshold_max_2048b", |bencher| {
            let mut out = vec![0u32; BATCH];
            bencher.iter(|| {
                array::batch_threshold(
                    black_box(&source),
                    black_box(&targets),
                    u32::MAX,
                    &mut out,
                );
                black_box(out[0])
            })
        });
    }

    group.finish();
}

criterion_group!(benches, single_comparison, batch_comparison);
criterion_main!(benches);

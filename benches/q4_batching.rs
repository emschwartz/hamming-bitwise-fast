//! Q4: Does batching help? Fixed-size batch vs variable-size batch?
//!
//! Key questions:
//! - How much faster is batch processing vs looping single calls?
//! - Does the compiler optimize fixed-size batches better?
//! - What's the overhead of allocation patterns?
//!
//! Run with: cargo bench --bench q4_batching
//! Filter by size: cargo bench --bench q4_batching -- 1024
//! Filter by batch: cargo bench --bench q4_batching -- batch_64

mod helpers;

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use hamming_bitwise_fast::{hamming, hamming_batch};
use helpers::*;
use std::cell::Cell;

fn batching_benchmarks(c: &mut Criterion) {
    // ========================================================================
    // Per-comparison time: cycling through targets of different batch sizes
    // ========================================================================
    {
        let mut group = c.benchmark_group("per_comparison");

        // Single pair (no batch context) - baseline
        macro_rules! bench_single {
            ($($bits:literal => $bytes:literal),+ $(,)?) => {
                $(
                    {
                        let a: [u8; $bytes] = random_bytes();
                        let b: [u8; $bytes] = random_bytes();
                        group.bench_function(
                            BenchmarkId::new("single_pair", concat!(stringify!($bits), "b")),
                            |bench| {
                                bench.iter(|| hamming(black_box(&a), black_box(&b)));
                            },
                        );
                    }
                )+
            };
        }
        bench_single!(512 => 64, 768 => 96, 1024 => 128, 2048 => 256);

        // One comparison from batch of 64 (fits in L1)
        macro_rules! bench_from_batch {
            ($batch_size:literal, $name:literal, $($bits:literal => $bytes:literal),+ $(,)?) => {
                $(
                    {
                        let source: [u8; $bytes] = random_bytes();
                        let targets: Vec<[u8; $bytes]> = random_bytes_array($batch_size);
                        let idx = Cell::new(0usize);

                        group.bench_function(
                            BenchmarkId::new($name, concat!(stringify!($bits), "b")),
                            |bench| {
                                bench.iter(|| {
                                    let i = idx.get();
                                    let result = hamming(black_box(&source), black_box(&targets[i]));
                                    idx.set((i + 1) % $batch_size);
                                    result
                                });
                            },
                        );
                    }
                )+
            };
        }
        bench_from_batch!(64, "batch_64", 512 => 64, 768 => 96, 1024 => 128, 2048 => 256);
        bench_from_batch!(256, "batch_256", 512 => 64, 768 => 96, 1024 => 128, 2048 => 256);
        bench_from_batch!(1024, "batch_1024", 512 => 64, 768 => 96, 1024 => 128, 2048 => 256);

        group.finish();
    }

    // ========================================================================
    // Batch API throughput
    // ========================================================================
    {
        let mut group = c.benchmark_group("batch_api");

        macro_rules! bench_batch_api {
            ($batch_size:literal, $name:literal, $($bits:literal => $bytes:literal),+ $(,)?) => {
                $(
                    {
                        let source: [u8; $bytes] = random_bytes();
                        let targets: Vec<[u8; $bytes]> = random_bytes_array($batch_size);
                        let mut out = vec![0u32; $batch_size];

                        group.throughput(Throughput::Elements($batch_size as u64));
                        group.bench_function(
                            BenchmarkId::new($name, concat!(stringify!($bits), "b")),
                            |bench| {
                                bench.iter(|| {
                                    hamming_batch(
                                        black_box(&source),
                                        black_box(&targets),
                                        black_box(&mut out),
                                    );
                                    black_box(out[0])
                                });
                            },
                        );
                    }
                )+
            };
        }
        bench_batch_api!(64, "hamming_batch_64", 512 => 64, 768 => 96, 1024 => 128, 2048 => 256);
        bench_batch_api!(256, "hamming_batch_256", 512 => 64, 768 => 96, 1024 => 128, 2048 => 256);

        group.finish();
    }

    // ========================================================================
    // Fixed vs variable batch size (64 elements)
    // ========================================================================
    {
        let mut group = c.benchmark_group("fixed_vs_variable");
        const BATCH: usize = 64;

        macro_rules! bench_variable {
            ($($bits:literal => $bytes:literal),+ $(,)?) => {
                $(
                    {
                        let source: [u8; $bytes] = random_bytes();
                        let targets: Vec<[u8; $bytes]> = random_bytes_array(BATCH);
                        let mut out = vec![0u32; BATCH];

                        group.throughput(Throughput::Elements(BATCH as u64));
                        group.bench_function(
                            BenchmarkId::new("variable", concat!(stringify!($bits), "b")),
                            |bench| {
                                bench.iter(|| {
                                    hamming_batch_variable(
                                        black_box(&source),
                                        black_box(&targets),
                                        black_box(&mut out),
                                    );
                                    black_box(out[0])
                                });
                            },
                        );
                    }
                )+
            };
        }
        bench_variable!(512 => 64, 768 => 96, 1024 => 128, 2048 => 256);

        macro_rules! bench_fixed {
            ($($bits:literal => $bytes:literal),+ $(,)?) => {
                $(
                    {
                        let source: [u8; $bytes] = random_bytes();
                        let targets_vec: Vec<[u8; $bytes]> = random_bytes_array(BATCH);
                        let targets: [[u8; $bytes]; BATCH] = targets_vec.try_into().unwrap();
                        let mut out = [0u32; BATCH];

                        group.throughput(Throughput::Elements(BATCH as u64));
                        group.bench_function(
                            BenchmarkId::new("fixed", concat!(stringify!($bits), "b")),
                            |bench| {
                                bench.iter(|| {
                                    hamming_batch_fixed(
                                        black_box(&source),
                                        black_box(&targets),
                                        black_box(&mut out),
                                    );
                                    black_box(out[0])
                                });
                            },
                        );
                    }
                )+
            };
        }
        bench_fixed!(512 => 64, 768 => 96, 1024 => 128, 2048 => 256);

        group.finish();
    }

    // ========================================================================
    // Allocation patterns (64 element batches)
    // ========================================================================
    {
        let mut group = c.benchmark_group("allocation");
        const BATCH: usize = 64;

        macro_rules! bench_alloc {
            ($($bits:literal => $bytes:literal),+ $(,)?) => {
                $(
                    {
                        let source: [u8; $bytes] = random_bytes();
                        let targets: Vec<[u8; $bytes]> = random_bytes_array(BATCH);

                        group.throughput(Throughput::Elements(BATCH as u64));
                        group.bench_function(
                            BenchmarkId::new("alloc_per_batch", concat!(stringify!($bits), "b")),
                            |bench| {
                                bench.iter(|| {
                                    let mut out = vec![0u32; BATCH];
                                    hamming_batch(
                                        black_box(&source),
                                        black_box(&targets),
                                        black_box(&mut out),
                                    );
                                    black_box(out[0])
                                });
                            },
                        );
                    }
                )+
            };
        }
        bench_alloc!(512 => 64, 768 => 96, 1024 => 128, 2048 => 256);

        macro_rules! bench_prealloc {
            ($($bits:literal => $bytes:literal),+ $(,)?) => {
                $(
                    {
                        let source: [u8; $bytes] = random_bytes();
                        let targets: Vec<[u8; $bytes]> = random_bytes_array(BATCH);
                        let mut out = vec![0u32; BATCH];

                        group.throughput(Throughput::Elements(BATCH as u64));
                        group.bench_function(
                            BenchmarkId::new("preallocated", concat!(stringify!($bits), "b")),
                            |bench| {
                                bench.iter(|| {
                                    hamming_batch(
                                        black_box(&source),
                                        black_box(&targets),
                                        black_box(&mut out),
                                    );
                                    black_box(out[0])
                                });
                            },
                        );
                    }
                )+
            };
        }
        bench_prealloc!(512 => 64, 768 => 96, 1024 => 128, 2048 => 256);

        group.finish();
    }
}

criterion_group!(benches, batching_benchmarks);
criterion_main!(benches);

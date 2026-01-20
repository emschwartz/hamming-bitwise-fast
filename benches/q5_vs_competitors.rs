//! Q5: How much better are our optimizations than the original and other crates?
//!
//! Key questions:
//! - How does our new hamming<N> compare to the original hamming_bitwise_fast?
//! - How do we compare to other Hamming distance crates (simsimd, hamming, triple_accel)?
//! - What's the speedup from all optimizations combined (batch + arrays)?
//!
//! Run with: cargo bench --bench q5_vs_competitors
//! With multiversion: cargo bench --features multiversion --bench q5_vs_competitors

mod helpers;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hamming_bitwise_fast::{
    hamming_bitwise_array, hamming_bitwise_array_batch, hamming_bitwise_slice,
};
use helpers::*;

// ============================================================================
// Single comparison benchmarks
// ============================================================================

fn single_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("single");

    for size in BIT_SIZES {
        let bytes = size.bytes();
        let a_vec = random_bytes_vec(bytes);
        let b_vec = random_bytes_vec(bytes);

        // Original hamming_bitwise_fast (slice API)
        group.bench_with_input(
            BenchmarkId::new("hamming_bitwise_slice", size),
            &size,
            |b, _| {
                b.iter(|| hamming_bitwise_slice(black_box(&a_vec), black_box(&b_vec)));
            },
        );

        // simsimd
        group.bench_with_input(BenchmarkId::new("simsimd", size), &size, |b, _| {
            b.iter(|| simsimd::BinarySimilarity::hamming(black_box(&a_vec), black_box(&b_vec)));
        });

        // hamming crate
        group.bench_with_input(BenchmarkId::new("hamming_crate", size), &size, |b, _| {
            b.iter(|| hamming::distance_fast(black_box(&a_vec), black_box(&b_vec)));
        });

        // triple_accel
        group.bench_with_input(BenchmarkId::new("triple_accel", size), &size, |b, _| {
            b.iter(|| triple_accel::hamming(black_box(&a_vec), black_box(&b_vec)));
        });

        // hamming_rs (x86 only)
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        group.bench_with_input(BenchmarkId::new("hamming_rs", size), &size, |b, _| {
            b.iter(|| hamming_rs::distance_faster(black_box(&a_vec), black_box(&b_vec)));
        });
    }

    // Our const-generic array API (u8 arrays)
    // N is in bytes: 64=512bit, 96=768bit, 128=1024bit, 256=2048bit
    macro_rules! bench_hamming_n {
        ($($bits:literal => $bytes:literal),+ $(,)?) => {
            $(
                {
                    let a: [u8; $bytes] = random_bytes();
                    let b: [u8; $bytes] = random_bytes();
                    group.bench_function(
                        BenchmarkId::new("hamming_bitwise_array", concat!(stringify!($bits), "b")),
                        |bench| {
                            bench.iter(|| hamming_bitwise_array(black_box(&a), black_box(&b)));
                        },
                    );
                }
            )+
        };
    }
    bench_hamming_n!(512 => 64, 768 => 96, 1024 => 128, 2048 => 256);

    group.finish();
}

// ============================================================================
// Batch comparison benchmarks (64 elements)
// ============================================================================

fn batch_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_64");
    const BATCH: usize = 64;

    // Our batch API with const-generic u8 arrays
    macro_rules! bench_batch {
        ($($bits:literal => $bytes:literal),+ $(,)?) => {
            $(
                {
                    let source: [u8; $bytes] = random_bytes();
                    let targets: Vec<[u8; $bytes]> = random_bytes_array(BATCH);
                    let mut out = vec![0u32; BATCH];

                    group.throughput(Throughput::Elements(BATCH as u64));
                    group.bench_function(
                        BenchmarkId::new("hamming_bitwise_array_batch", concat!(stringify!($bits), "b")),
                        |bench| {
                            bench.iter(|| {
                                hamming_bitwise_array_batch(
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
    bench_batch!(512 => 64, 768 => 96, 1024 => 128, 2048 => 256);

    // Our hamming<N> in a loop
    macro_rules! bench_loop {
        ($($bits:literal => $bytes:literal),+ $(,)?) => {
            $(
                {
                    let source: [u8; $bytes] = random_bytes();
                    let targets: Vec<[u8; $bytes]> = random_bytes_array(BATCH);
                    let mut out = vec![0u32; BATCH];

                    group.throughput(Throughput::Elements(BATCH as u64));
                    group.bench_function(
                        BenchmarkId::new("hamming_bitwise_array_loop", concat!(stringify!($bits), "b")),
                        |bench| {
                            bench.iter(|| {
                                for (i, target) in targets.iter().enumerate() {
                                    out[i] = hamming_bitwise_array(black_box(&source), black_box(target));
                                }
                                black_box(out[0])
                            });
                        },
                    );
                }
            )+
        };
    }
    bench_loop!(512 => 64, 768 => 96, 1024 => 128, 2048 => 256);

    // Competitor loops with slice APIs
    for size in BIT_SIZES {
        let bytes = size.bytes();
        let source = random_bytes_vec(bytes);
        let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(bytes)).collect();

        group.throughput(Throughput::Elements(BATCH as u64));

        // Original hamming_bitwise_fast loop
        {
            let mut out = vec![0u32; BATCH];
            group.bench_with_input(
                BenchmarkId::new("hamming_bitwise_slice_loop", size),
                &size,
                |b, _| {
                    b.iter(|| {
                        for (i, target) in targets.iter().enumerate() {
                            out[i] = hamming_bitwise_slice(black_box(&source), black_box(target));
                        }
                        black_box(out[0])
                    });
                },
            );
        }

        // simsimd loop
        {
            let mut out = vec![0f64; BATCH];
            group.bench_with_input(BenchmarkId::new("simsimd_loop", size), &size, |b, _| {
                b.iter(|| {
                    for (i, target) in targets.iter().enumerate() {
                        out[i] = simsimd::BinarySimilarity::hamming(
                            black_box(&source),
                            black_box(target),
                        )
                        .unwrap_or(0.0);
                    }
                    black_box(out[0] as u64)
                });
            });
        }

        // triple_accel loop
        {
            let mut out = vec![0u32; BATCH];
            group.bench_with_input(
                BenchmarkId::new("triple_accel_loop", size),
                &size,
                |b, _| {
                    b.iter(|| {
                        for (i, target) in targets.iter().enumerate() {
                            out[i] = triple_accel::hamming(black_box(&source), black_box(target));
                        }
                        black_box(out[0])
                    });
                },
            );
        }

        // hamming crate loop
        {
            let mut out = vec![0u64; BATCH];
            group.bench_with_input(
                BenchmarkId::new("hamming_crate_loop", size),
                &size,
                |b, _| {
                    b.iter(|| {
                        for (i, target) in targets.iter().enumerate() {
                            out[i] = hamming::distance_fast(black_box(&source), black_box(target))
                                .unwrap_or(0);
                        }
                        black_box(out[0])
                    });
                },
            );
        }

        // hamming_rs loop (x86 only)
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let mut out = vec![0u64; BATCH];
            group.bench_with_input(BenchmarkId::new("hamming_rs_loop", size), &size, |b, _| {
                b.iter(|| {
                    for (i, target) in targets.iter().enumerate() {
                        out[i] = hamming_rs::distance_faster(black_box(&source), black_box(target));
                    }
                    black_box(out[0])
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, single_comparison, batch_comparison);
criterion_main!(benches);

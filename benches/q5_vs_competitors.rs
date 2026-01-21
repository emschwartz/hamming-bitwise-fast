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

use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
};
use std::hint::black_box;
use hamming_bitwise_fast::{
    hamming_bitwise_array, hamming_bitwise_array_batch, hamming_bitwise_slice,
    hamming_bitwise_slice_batch,
};
use helpers::*;

// ============================================================================
// Single comparison benchmarks
// ============================================================================

fn single_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("single");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

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

    for size in BIT_SIZES {
        let bytes = size.bytes();
        let a_vec = random_bytes_vec(bytes);
        let b_vec = random_bytes_vec(bytes);

        // Current hamming_bitwise_slice (with multiversion when enabled)
        group.bench_with_input(
            BenchmarkId::new("hamming_bitwise_slice", size),
            &size,
            |b, _| {
                b.iter(|| hamming_bitwise_slice(black_box(&a_vec), black_box(&b_vec)));
            },
        );

        // Original v1 implementation (no multiversion, u64 chunked)
        group.bench_with_input(
            BenchmarkId::new("hamming_bitwise_slice_v1", size),
            &size,
            |b, _| {
                b.iter(|| hamming_bitwise_slice_v1(black_box(&a_vec), black_box(&b_vec)));
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

    group.finish();
}

// ============================================================================
// Batch comparison benchmarks (1000 comparisons per iteration, divide time by 1000)
// ============================================================================

fn batch_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_1000");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    const BATCH: usize = 1000;

    // Our batch API with const-generic u8 arrays
    macro_rules! bench_batch_array {
        ($($bits:literal => $bytes:literal),+ $(,)?) => {
            $(
                {
                    let source: [u8; $bytes] = random_bytes();
                    let targets: Vec<[u8; $bytes]> = random_bytes_array(BATCH);
                    let mut out = vec![0u32; BATCH];

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
    bench_batch_array!(512 => 64, 768 => 96, 1024 => 128, 2048 => 256);

    // Our hamming_bitwise_array in a loop (1000 comparisons)
    macro_rules! bench_array_loop {
        ($($bits:literal => $bytes:literal),+ $(,)?) => {
            $(
                {
                    let source: [u8; $bytes] = random_bytes();
                    let targets: Vec<[u8; $bytes]> = random_bytes_array(BATCH);
                    let mut out = vec![0u32; BATCH];

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
    bench_array_loop!(512 => 64, 768 => 96, 1024 => 128, 2048 => 256);

    // Competitor benchmarks (1000 comparisons per iteration)
    for size in BIT_SIZES {
        let bytes = size.bytes();
        let source = random_bytes_vec(bytes);
        let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(bytes)).collect();
        let targets_refs: Vec<&[u8]> = targets.iter().map(|v| v.as_slice()).collect();

        // Our batch slice API
        {
            let mut out = vec![0u32; BATCH];
            group.bench_with_input(
                BenchmarkId::new("hamming_bitwise_slice_batch", size),
                &size,
                |b, _| {
                    b.iter(|| {
                        hamming_bitwise_slice_batch(
                            black_box(&source),
                            black_box(&targets_refs),
                            black_box(&mut out),
                        );
                        black_box(out[0])
                    });
                },
            );
        }

        // hamming_bitwise_slice in a loop
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

        // Original v1 implementation in a loop
        {
            let mut out = vec![0u32; BATCH];
            group.bench_with_input(
                BenchmarkId::new("hamming_bitwise_slice_v1_loop", size),
                &size,
                |b, _| {
                    b.iter(|| {
                        for (i, target) in targets.iter().enumerate() {
                            out[i] = hamming_bitwise_slice_v1(black_box(&source), black_box(target));
                        }
                        black_box(out[0])
                    });
                },
            );
        }

        // simsimd in a loop
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

        // hamming crate in a loop
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

        // triple_accel in a loop
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

        // hamming_rs in a loop (x86 only)
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

//! Q7: How does our batch slice API compare to individual slice calls and competitors?
//!
//! Key questions:
//! - Does hamming_bitwise_slice_batch provide speedup over hamming_bitwise_slice in a loop?
//! - How does our batch slice compare to competitor crates in loops?
//!
//! Run with: cargo bench --bench q7_batch_slice
//! With multiversion: cargo bench --features multiversion_x86 --bench q7_batch_slice

mod helpers;

use criterion::{black_box, criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration, Throughput};
use hamming_bitwise_fast::{hamming_bitwise_slice, hamming_bitwise_slice_batch};
use helpers::*;

const BATCH: usize = 64;

// ============================================================================
// Group 1: Our batch slice API vs our slice in a loop
// ============================================================================

fn batch_slice_vs_loop(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_slice_vs_loop");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Linear));

    for size in BIT_SIZES {
        let bytes = size.bytes();

        // Pre-allocate all data before benchmarks
        let source = random_bytes_vec(bytes);
        let targets_owned: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(bytes)).collect();
        let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();

        group.throughput(Throughput::Elements(BATCH as u64));

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
                            black_box(&targets),
                            black_box(&mut out),
                        );
                        black_box(out[0])
                    });
                },
            );
        }

        // Our slice API in a loop
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
    }

    group.finish();
}

// ============================================================================
// Group 2: Head-to-head comparison with competitors
// ============================================================================

fn batch_slice_competitors(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_slice_competitors");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Linear));

    for size in BIT_SIZES {
        let bytes = size.bytes();

        // Pre-allocate all data before benchmarks
        let source = random_bytes_vec(bytes);
        let targets_owned: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(bytes)).collect();
        let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();

        group.throughput(Throughput::Elements(BATCH as u64));

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
                            black_box(&targets),
                            black_box(&mut out),
                        );
                        black_box(out[0])
                    });
                },
            );
        }

        // Our slice API in a loop
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
                    for (i, target) in targets_owned.iter().enumerate() {
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

        // hamming crate loop
        {
            let mut out = vec![0u64; BATCH];
            group.bench_with_input(
                BenchmarkId::new("hamming_crate_loop", size),
                &size,
                |b, _| {
                    b.iter(|| {
                        for (i, target) in targets_owned.iter().enumerate() {
                            out[i] = hamming::distance_fast(black_box(&source), black_box(target))
                                .unwrap_or(0);
                        }
                        black_box(out[0])
                    });
                },
            );
        }

        // triple_accel loop
        {
            let mut out = vec![0u32; BATCH];
            group.bench_with_input(
                BenchmarkId::new("triple_accel_loop", size),
                &size,
                |b, _| {
                    b.iter(|| {
                        for (i, target) in targets_owned.iter().enumerate() {
                            out[i] = triple_accel::hamming(black_box(&source), black_box(target));
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
                    for (i, target) in targets_owned.iter().enumerate() {
                        out[i] = hamming_rs::distance_faster(black_box(&source), black_box(target));
                    }
                    black_box(out[0])
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, batch_slice_vs_loop, batch_slice_competitors);
criterion_main!(benches);

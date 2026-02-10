//! Compares this crate vs other Hamming distance implementations.
//!
//! Competitors:
//! - simsimd: SIMD-optimized similarity functions
//! - hamming: Pure Rust implementation
//! - triple_accel: SIMD-accelerated string metrics
//! - hamming_rs: x86-only AVX2/SSE implementation
//!
//! ## Methodology
//!
//! Single benchmarks use `iter_batched_ref` to generate fresh random inputs
//! per batch. This avoids `black_box`-on-inputs, which would create an unfair
//! asymmetry: `black_box` hides the fat-pointer length for slices but can't
//! hide the const generic `N` baked into array types. With `iter_batched_ref`,
//! all contestants receive the same `&mut [u8; N]` data and the same type
//! information — any advantage from `#[inline]` + known sizes is a legitimate
//! advantage that users get in real code.
//!
//! Run with: cargo criterion --bench competitors
//! Quick mode: cargo criterion --bench competitors -- --quick

mod helpers;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use helpers::{l1_batch_size, random_bytes, random_bytes_array};

use std::hint::black_box;

const BATCH: usize = 1000;

// ============================================================================
// Baseline v1: Simple byte-by-byte iteration, no architecture targeting
// ============================================================================

/// Original v1 implementation - u64 chunking with remainder, no architecture targeting.
/// This is what auto-vectorization gives us without multiversion or target-cpu flags.
#[inline]
fn hamming_v1(x: &[u8], y: &[u8]) -> u32 {
    assert_eq!(x.len(), y.len());

    // Process 8 bytes at a time using u64
    let mut distance = x
        .chunks_exact(8)
        .zip(y.chunks_exact(8))
        .map(|(x_chunk, y_chunk)| {
            let x_val = u64::from_ne_bytes(x_chunk.try_into().unwrap());
            let y_val = u64::from_ne_bytes(y_chunk.try_into().unwrap());
            (x_val ^ y_val).count_ones()
        })
        .sum::<u32>();

    if x.len() % 8 != 0 {
        distance += x
            .chunks_exact(8)
            .remainder()
            .iter()
            .zip(y.chunks_exact(8).remainder())
            .map(|(x_byte, y_byte)| (x_byte ^ y_byte).count_ones())
            .sum::<u32>();
    }

    distance
}

// ============================================================================
// Single comparison benchmarks
// ============================================================================

fn single_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("single");

    macro_rules! bench_size {
        ($size:expr) => {{
            let bits = $size * 8;
            group.throughput(Throughput::Elements(1));

            let setup = || (random_bytes::<$size>(), random_bytes::<$size>());

            group.bench_function(BenchmarkId::new("array::distance", bits), |b| {
                b.iter_batched_ref(
                    setup,
                    |(a, b)| hamming_bitwise_fast::array::distance(a, b),
                    l1_batch_size(2 * $size),
                )
            });

            group.bench_function(BenchmarkId::new("slice::distance", bits), |b| {
                b.iter_batched_ref(
                    setup,
                    |(a, b)| hamming_bitwise_fast::slice::distance(a.as_slice(), b.as_slice()),
                    l1_batch_size(2 * $size),
                )
            });

            group.bench_function(BenchmarkId::new("hamming_bitwise_fast_v1", bits), |b| {
                b.iter_batched_ref(
                    setup,
                    |(a, b)| hamming_v1(a.as_slice(), b.as_slice()),
                    l1_batch_size(2 * $size),
                )
            });

            // simsimd — returns a normalized f64 (distance / total_bits), so it
            // performs additional floating-point division vs raw integer implementations.
            group.bench_function(BenchmarkId::new("simsimd", bits), |b| {
                b.iter_batched_ref(
                    setup,
                    |(a, b)| simsimd::BinarySimilarity::hamming(a.as_slice(), b.as_slice()),
                    l1_batch_size(2 * $size),
                )
            });

            group.bench_function(BenchmarkId::new("hamming_crate", bits), |b| {
                b.iter_batched_ref(
                    setup,
                    |(a, b)| hamming::distance_fast(a.as_slice(), b.as_slice()),
                    l1_batch_size(2 * $size),
                )
            });

            group.bench_function(BenchmarkId::new("triple_accel", bits), |b| {
                b.iter_batched_ref(
                    setup,
                    |(a, b)| triple_accel::hamming(a.as_slice(), b.as_slice()),
                    l1_batch_size(2 * $size),
                )
            });

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            group.bench_function(BenchmarkId::new("hamming_rs", bits), |b| {
                b.iter_batched_ref(
                    setup,
                    |(a, b)| hamming_rs::distance_faster(a.as_slice(), b.as_slice()),
                    l1_batch_size(2 * $size),
                )
            });
        }};
    }

    bench_size!(64);
    bench_size!(96);
    bench_size!(128);
    bench_size!(256);

    group.finish();
}

// ============================================================================
// Batch comparison benchmarks (1000 comparisons)
// ============================================================================

fn batch_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch");

    macro_rules! bench_size {
        ($size:expr) => {{
            let bits = $size * 8;
            group.throughput(Throughput::Elements(BATCH as u64));

            let source: [u8; $size] = random_bytes();
            let targets: Vec<[u8; $size]> = random_bytes_array(BATCH);
            let source_vec = source.to_vec();
            let targets_vecs: Vec<Vec<u8>> = targets.iter().map(|t| t.to_vec()).collect();
            let targets_refs: Vec<&[u8]> = targets_vecs.iter().map(|v| v.as_slice()).collect();
            let mut out = vec![0u32; BATCH];

            group.bench_with_input(
                BenchmarkId::new("array::batch", bits),
                &$size,
                |bencher, _| {
                    bencher.iter(|| {
                        hamming_bitwise_fast::array::batch(
                            black_box(&source),
                            black_box(&targets),
                            black_box(&mut out),
                        );
                        black_box(out[0])
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new("slice::batch", bits),
                &$size,
                |bencher, _| {
                    bencher.iter(|| {
                        hamming_bitwise_fast::slice::batch(
                            black_box(&source_vec),
                            black_box(&targets_refs),
                            black_box(&mut out),
                        );
                        black_box(out[0])
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new("slice::distance (loop)", bits),
                &$size,
                |bencher, _| {
                    bencher.iter(|| {
                        for (i, target) in black_box(&targets).iter().enumerate() {
                            out[i] = hamming_bitwise_fast::slice::distance(
                                black_box(&source_vec),
                                target.as_slice(),
                            );
                        }
                        black_box(out[0])
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new("hamming_bitwise_fast_v1", bits),
                &$size,
                |bencher, _| {
                    bencher.iter(|| {
                        for (i, target) in black_box(&targets_vecs).iter().enumerate() {
                            out[i] = hamming_v1(black_box(&source_vec), target);
                        }
                        black_box(out[0])
                    })
                },
            );

            // simsimd batch — returns a normalized f64 (distance / total_bits), so it
            // performs additional floating-point division vs raw integer implementations.
            {
                let mut out_f64 = vec![0f64; BATCH];
                group.bench_with_input(BenchmarkId::new("simsimd", bits), &$size, |bencher, _| {
                    bencher.iter(|| {
                        for (i, target) in black_box(&targets_vecs).iter().enumerate() {
                            out_f64[i] =
                                simsimd::BinarySimilarity::hamming(black_box(&source_vec), target)
                                    .unwrap_or(0.0);
                        }
                        black_box(out_f64[0] as u64)
                    })
                });
            }

            {
                let mut out_u64 = vec![0u64; BATCH];
                group.bench_with_input(
                    BenchmarkId::new("hamming_crate", bits),
                    &$size,
                    |bencher, _| {
                        bencher.iter(|| {
                            for (i, target) in black_box(&targets_vecs).iter().enumerate() {
                                out_u64[i] = hamming::distance_fast(black_box(&source_vec), target)
                                    .unwrap_or(0);
                            }
                            black_box(out_u64[0])
                        })
                    },
                );
            }

            group.bench_with_input(
                BenchmarkId::new("triple_accel", bits),
                &$size,
                |bencher, _| {
                    bencher.iter(|| {
                        for (i, target) in black_box(&targets_vecs).iter().enumerate() {
                            out[i] = triple_accel::hamming(black_box(&source_vec), target);
                        }
                        black_box(out[0])
                    })
                },
            );

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                let mut out_u64 = vec![0u64; BATCH];
                group.bench_with_input(
                    BenchmarkId::new("hamming_rs", bits),
                    &$size,
                    |bencher, _| {
                        bencher.iter(|| {
                            for (i, target) in black_box(&targets_vecs).iter().enumerate() {
                                out_u64[i] =
                                    hamming_rs::distance_faster(black_box(&source_vec), target);
                            }
                            black_box(out_u64[0])
                        })
                    },
                );
            }
        }};
    }

    bench_size!(64);
    bench_size!(96);
    bench_size!(128);
    bench_size!(256);

    group.finish();
}

criterion_group!(benches, single_benchmarks, batch_benchmarks,);
criterion_main!(benches);

//! Compares this crate vs other Hamming distance implementations.
//!
//! Competitors:
//! - simsimd: SIMD-optimized similarity functions
//! - hamming: Pure Rust implementation
//! - triple_accel: SIMD-accelerated string metrics
//! - hamming_rs: x86-only AVX2/SSE implementation
//!
//! Run with: cargo bench --bench competitors
//! Quick mode: cargo bench --bench competitors -- --quick

mod helpers;

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use helpers::{random_bytes, random_bytes_array, random_bytes_vec};

const BATCH: usize = 1000;
const SIZES: &[usize] = &[64, 96, 128, 256];

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

    for &size in SIZES {
        group.throughput(Throughput::Bits((size * 8 * 2) as u64));

        // hamming_bitwise_array - requires const generics, so we dispatch manually
        match size {
            64 => {
                let a: [u8; 64] = random_bytes();
                let b: [u8; 64] = random_bytes();
                group.bench_with_input(
                    BenchmarkId::new("hamming_bitwise_array", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::hamming_bitwise_array(
                                black_box(&a),
                                black_box(&b),
                            ))
                        })
                    },
                );
            }
            96 => {
                let a: [u8; 96] = random_bytes();
                let b: [u8; 96] = random_bytes();
                group.bench_with_input(
                    BenchmarkId::new("hamming_bitwise_array", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::hamming_bitwise_array(
                                black_box(&a),
                                black_box(&b),
                            ))
                        })
                    },
                );
            }
            128 => {
                let a: [u8; 128] = random_bytes();
                let b: [u8; 128] = random_bytes();
                group.bench_with_input(
                    BenchmarkId::new("hamming_bitwise_array", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::hamming_bitwise_array(
                                black_box(&a),
                                black_box(&b),
                            ))
                        })
                    },
                );
            }
            256 => {
                let a: [u8; 256] = random_bytes();
                let b: [u8; 256] = random_bytes();
                group.bench_with_input(
                    BenchmarkId::new("hamming_bitwise_array", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::hamming_bitwise_array(
                                black_box(&a),
                                black_box(&b),
                            ))
                        })
                    },
                );
            }
            _ => {}
        }

        // hamming_bitwise_slice
        {
            let a = random_bytes_vec(size);
            let b = random_bytes_vec(size);
            group.bench_with_input(
                BenchmarkId::new("hamming_bitwise_slice", format!("{}b", size * 8)),
                &size,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(hamming_bitwise_fast::hamming_bitwise_slice(
                            black_box(&a),
                            black_box(&b),
                        ))
                    })
                },
            );
        }

        // hamming_bitwise_fast v1 (baseline - no arch targeting)
        {
            let a = random_bytes_vec(size);
            let b = random_bytes_vec(size);
            group.bench_with_input(
                BenchmarkId::new("hamming_bitwise_fast_v1", format!("{}b", size * 8)),
                &size,
                |bencher, _| {
                    bencher.iter(|| black_box(hamming_v1(black_box(&a), black_box(&b))))
                },
            );
        }

        // simsimd
        {
            let a = random_bytes_vec(size);
            let b = random_bytes_vec(size);
            group.bench_with_input(BenchmarkId::new("simsimd", format!("{}b", size * 8)), &size, |bencher, _| {
                bencher.iter(|| {
                    black_box(simsimd::BinarySimilarity::hamming(
                        black_box(&a),
                        black_box(&b),
                    ))
                })
            });
        }

        // hamming crate
        {
            let a = random_bytes_vec(size);
            let b = random_bytes_vec(size);
            group.bench_with_input(
                BenchmarkId::new("hamming_crate", format!("{}b", size * 8)),
                &size,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(hamming::distance_fast(black_box(&a), black_box(&b)))
                    })
                },
            );
        }

        // triple_accel
        {
            let a = random_bytes_vec(size);
            let b = random_bytes_vec(size);
            group.bench_with_input(
                BenchmarkId::new("triple_accel", format!("{}b", size * 8)),
                &size,
                |bencher, _| {
                    bencher.iter(|| black_box(triple_accel::hamming(black_box(&a), black_box(&b))))
                },
            );
        }

        // hamming_rs (x86 only)
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let a = random_bytes_vec(size);
            let b = random_bytes_vec(size);
            group.bench_with_input(
                BenchmarkId::new("hamming_rs", format!("{}b", size * 8)),
                &size,
                |bencher, _| {
                    bencher
                        .iter(|| black_box(hamming_rs::distance_faster(black_box(&a), black_box(&b))))
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// Batch comparison benchmarks (1000 comparisons)
// ============================================================================

fn batch_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch");

    for &size in SIZES {
        group.throughput(Throughput::Bits((size * 8 * 2 * BATCH) as u64));

        // hamming_bitwise_array_batch - requires const generics
        match size {
            64 => {
                let source: [u8; 64] = random_bytes();
                let targets: Vec<[u8; 64]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];
                group.bench_with_input(
                    BenchmarkId::new("hamming_bitwise_array_batch", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            hamming_bitwise_fast::hamming_bitwise_array_batch(
                                black_box(&source),
                                black_box(&targets),
                                black_box(&mut out),
                            );
                            black_box(out[0])
                        })
                    },
                );
            }
            96 => {
                let source: [u8; 96] = random_bytes();
                let targets: Vec<[u8; 96]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];
                group.bench_with_input(
                    BenchmarkId::new("hamming_bitwise_array_batch", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            hamming_bitwise_fast::hamming_bitwise_array_batch(
                                black_box(&source),
                                black_box(&targets),
                                black_box(&mut out),
                            );
                            black_box(out[0])
                        })
                    },
                );
            }
            128 => {
                let source: [u8; 128] = random_bytes();
                let targets: Vec<[u8; 128]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];
                group.bench_with_input(
                    BenchmarkId::new("hamming_bitwise_array_batch", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            hamming_bitwise_fast::hamming_bitwise_array_batch(
                                black_box(&source),
                                black_box(&targets),
                                black_box(&mut out),
                            );
                            black_box(out[0])
                        })
                    },
                );
            }
            256 => {
                let source: [u8; 256] = random_bytes();
                let targets: Vec<[u8; 256]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];
                group.bench_with_input(
                    BenchmarkId::new("hamming_bitwise_array_batch", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            hamming_bitwise_fast::hamming_bitwise_array_batch(
                                black_box(&source),
                                black_box(&targets),
                                black_box(&mut out),
                            );
                            black_box(out[0])
                        })
                    },
                );
            }
            _ => {}
        }

        // hamming_bitwise_array_loop - requires const generics
        match size {
            64 => {
                let source: [u8; 64] = random_bytes();
                let targets: Vec<[u8; 64]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];
                group.bench_with_input(
                    BenchmarkId::new("hamming_bitwise_array_loop", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            for (i, target) in black_box(&targets).iter().enumerate() {
                                out[i] = hamming_bitwise_fast::hamming_bitwise_array(
                                    black_box(&source),
                                    target,
                                );
                            }
                            black_box(out[0])
                        })
                    },
                );
            }
            96 => {
                let source: [u8; 96] = random_bytes();
                let targets: Vec<[u8; 96]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];
                group.bench_with_input(
                    BenchmarkId::new("hamming_bitwise_array_loop", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            for (i, target) in black_box(&targets).iter().enumerate() {
                                out[i] = hamming_bitwise_fast::hamming_bitwise_array(
                                    black_box(&source),
                                    target,
                                );
                            }
                            black_box(out[0])
                        })
                    },
                );
            }
            128 => {
                let source: [u8; 128] = random_bytes();
                let targets: Vec<[u8; 128]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];
                group.bench_with_input(
                    BenchmarkId::new("hamming_bitwise_array_loop", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            for (i, target) in black_box(&targets).iter().enumerate() {
                                out[i] = hamming_bitwise_fast::hamming_bitwise_array(
                                    black_box(&source),
                                    target,
                                );
                            }
                            black_box(out[0])
                        })
                    },
                );
            }
            256 => {
                let source: [u8; 256] = random_bytes();
                let targets: Vec<[u8; 256]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];
                group.bench_with_input(
                    BenchmarkId::new("hamming_bitwise_array_loop", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            for (i, target) in black_box(&targets).iter().enumerate() {
                                out[i] = hamming_bitwise_fast::hamming_bitwise_array(
                                    black_box(&source),
                                    target,
                                );
                            }
                            black_box(out[0])
                        })
                    },
                );
            }
            _ => {}
        }

        // hamming_bitwise_slice_batch
        {
            let source = random_bytes_vec(size);
            let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(size)).collect();
            let targets_refs: Vec<&[u8]> = targets.iter().map(|v| v.as_slice()).collect();
            let mut out = vec![0u32; BATCH];

            group.bench_with_input(
                BenchmarkId::new("hamming_bitwise_slice_batch", format!("{}b", size * 8)),
                &size,
                |bencher, _| {
                    bencher.iter(|| {
                        hamming_bitwise_fast::hamming_bitwise_slice_batch(
                            black_box(&source),
                            black_box(&targets_refs),
                            black_box(&mut out),
                        );
                        black_box(out[0])
                    })
                },
            );
        }

        // hamming_bitwise_slice_loop
        {
            let source = random_bytes_vec(size);
            let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(size)).collect();
            let mut out = vec![0u32; BATCH];

            group.bench_with_input(
                BenchmarkId::new("hamming_bitwise_slice_loop", format!("{}b", size * 8)),
                &size,
                |bencher, _| {
                    bencher.iter(|| {
                        for (i, target) in black_box(&targets).iter().enumerate() {
                            out[i] = hamming_bitwise_fast::hamming_bitwise_slice(
                                black_box(&source),
                                target,
                            );
                        }
                        black_box(out[0])
                    })
                },
            );
        }

        // hamming_bitwise_fast_v1 batch (baseline - no arch targeting)
        {
            let source = random_bytes_vec(size);
            let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(size)).collect();
            let mut out = vec![0u32; BATCH];

            group.bench_with_input(
                BenchmarkId::new("hamming_bitwise_fast_v1", format!("{}b", size * 8)),
                &size,
                |bencher, _| {
                    bencher.iter(|| {
                        for (i, target) in black_box(&targets).iter().enumerate() {
                            out[i] = hamming_v1(black_box(&source), target);
                        }
                        black_box(out[0])
                    })
                },
            );
        }

        // simsimd batch
        {
            let source = random_bytes_vec(size);
            let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(size)).collect();
            let mut out = vec![0f64; BATCH];

            group.bench_with_input(BenchmarkId::new("simsimd", format!("{}b", size * 8)), &size, |bencher, _| {
                bencher.iter(|| {
                    for (i, target) in black_box(&targets).iter().enumerate() {
                        out[i] = simsimd::BinarySimilarity::hamming(black_box(&source), target)
                            .unwrap_or(0.0);
                    }
                    black_box(out[0] as u64)
                })
            });
        }

        // hamming crate batch
        {
            let source = random_bytes_vec(size);
            let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(size)).collect();
            let mut out = vec![0u64; BATCH];

            group.bench_with_input(
                BenchmarkId::new("hamming_crate", format!("{}b", size * 8)),
                &size,
                |bencher, _| {
                    bencher.iter(|| {
                        for (i, target) in black_box(&targets).iter().enumerate() {
                            out[i] =
                                hamming::distance_fast(black_box(&source), target).unwrap_or(0);
                        }
                        black_box(out[0])
                    })
                },
            );
        }

        // triple_accel batch
        {
            let source = random_bytes_vec(size);
            let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(size)).collect();
            let mut out = vec![0u32; BATCH];

            group.bench_with_input(
                BenchmarkId::new("triple_accel", format!("{}b", size * 8)),
                &size,
                |bencher, _| {
                    bencher.iter(|| {
                        for (i, target) in black_box(&targets).iter().enumerate() {
                            out[i] = triple_accel::hamming(black_box(&source), target);
                        }
                        black_box(out[0])
                    })
                },
            );
        }

        // hamming_rs batch (x86 only)
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let source = random_bytes_vec(size);
            let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(size)).collect();
            let mut out = vec![0u64; BATCH];

            group.bench_with_input(
                BenchmarkId::new("hamming_rs", format!("{}b", size * 8)),
                &size,
                |bencher, _| {
                    bencher.iter(|| {
                        for (i, target) in black_box(&targets).iter().enumerate() {
                            out[i] = hamming_rs::distance_faster(black_box(&source), target);
                        }
                        black_box(out[0])
                    })
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, single_benchmarks, batch_benchmarks);
criterion_main!(benches);

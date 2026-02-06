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

        // array::distance - requires const generics, so we dispatch manually
        match size {
            64 => {
                let a: [u8; 64] = random_bytes();
                let b: [u8; 64] = random_bytes();
                group.bench_with_input(
                    BenchmarkId::new("array::distance", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::array::distance(
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
                    BenchmarkId::new("array::distance", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::array::distance(
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
                    BenchmarkId::new("array::distance", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::array::distance(
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
                    BenchmarkId::new("array::distance", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::array::distance(
                                black_box(&a),
                                black_box(&b),
                            ))
                        })
                    },
                );
            }
            _ => {}
        }

        // slice::distance
        {
            let a = random_bytes_vec(size);
            let b = random_bytes_vec(size);
            group.bench_with_input(
                BenchmarkId::new("slice::distance", format!("{}b", size * 8)),
                &size,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(hamming_bitwise_fast::slice::distance(
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

        // array::batch - requires const generics
        match size {
            64 => {
                let source: [u8; 64] = random_bytes();
                let targets: Vec<[u8; 64]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];
                group.bench_with_input(
                    BenchmarkId::new("array::batch", format!("{}b", size * 8)),
                    &size,
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
            }
            96 => {
                let source: [u8; 96] = random_bytes();
                let targets: Vec<[u8; 96]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];
                group.bench_with_input(
                    BenchmarkId::new("array::batch", format!("{}b", size * 8)),
                    &size,
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
            }
            128 => {
                let source: [u8; 128] = random_bytes();
                let targets: Vec<[u8; 128]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];
                group.bench_with_input(
                    BenchmarkId::new("array::batch", format!("{}b", size * 8)),
                    &size,
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
            }
            256 => {
                let source: [u8; 256] = random_bytes();
                let targets: Vec<[u8; 256]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];
                group.bench_with_input(
                    BenchmarkId::new("array::batch", format!("{}b", size * 8)),
                    &size,
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
            }
            _ => {}
        }

        // array::distance (loop) - requires const generics
        match size {
            64 => {
                let source: [u8; 64] = random_bytes();
                let targets: Vec<[u8; 64]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];
                group.bench_with_input(
                    BenchmarkId::new("array::distance (loop)", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            for (i, target) in black_box(&targets).iter().enumerate() {
                                out[i] = hamming_bitwise_fast::array::distance(
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
                    BenchmarkId::new("array::distance (loop)", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            for (i, target) in black_box(&targets).iter().enumerate() {
                                out[i] = hamming_bitwise_fast::array::distance(
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
                    BenchmarkId::new("array::distance (loop)", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            for (i, target) in black_box(&targets).iter().enumerate() {
                                out[i] = hamming_bitwise_fast::array::distance(
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
                    BenchmarkId::new("array::distance (loop)", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            for (i, target) in black_box(&targets).iter().enumerate() {
                                out[i] = hamming_bitwise_fast::array::distance(
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

        // slice::batch
        {
            let source = random_bytes_vec(size);
            let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(size)).collect();
            let targets_refs: Vec<&[u8]> = targets.iter().map(|v| v.as_slice()).collect();
            let mut out = vec![0u32; BATCH];

            group.bench_with_input(
                BenchmarkId::new("slice::batch", format!("{}b", size * 8)),
                &size,
                |bencher, _| {
                    bencher.iter(|| {
                        hamming_bitwise_fast::slice::batch(
                            black_box(&source),
                            black_box(&targets_refs),
                            black_box(&mut out),
                        );
                        black_box(out[0])
                    })
                },
            );
        }

        // slice::distance (loop)
        {
            let source = random_bytes_vec(size);
            let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(size)).collect();
            let mut out = vec![0u32; BATCH];

            group.bench_with_input(
                BenchmarkId::new("slice::distance (loop)", format!("{}b", size * 8)),
                &size,
                |bencher, _| {
                    bencher.iter(|| {
                        for (i, target) in black_box(&targets).iter().enumerate() {
                            out[i] = hamming_bitwise_fast::slice::distance(
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

// ============================================================================
// Threshold benchmarks (single comparison with early exit)
// ============================================================================

/// Threshold set to ~10% of max possible distance, so random vectors (which
/// average ~50% distance) are rejected early after the first block.
fn threshold_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("threshold");

    for &size in SIZES {
        let max = (size * 8) as u32 / 10; // ~10% threshold → most random pairs rejected early
        group.throughput(Throughput::Bits((size * 8 * 2) as u64));

        // array::threshold - requires const generics
        match size {
            64 => {
                let a: [u8; 64] = random_bytes();
                let b: [u8; 64] = random_bytes();
                group.bench_with_input(
                    BenchmarkId::new("array::threshold", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::array::threshold(
                                black_box(&a),
                                black_box(&b),
                                black_box(max),
                            ))
                        })
                    },
                );
            }
            96 => {
                let a: [u8; 96] = random_bytes();
                let b: [u8; 96] = random_bytes();
                group.bench_with_input(
                    BenchmarkId::new("array::threshold", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::array::threshold(
                                black_box(&a),
                                black_box(&b),
                                black_box(max),
                            ))
                        })
                    },
                );
            }
            128 => {
                let a: [u8; 128] = random_bytes();
                let b: [u8; 128] = random_bytes();
                group.bench_with_input(
                    BenchmarkId::new("array::threshold", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::array::threshold(
                                black_box(&a),
                                black_box(&b),
                                black_box(max),
                            ))
                        })
                    },
                );
            }
            256 => {
                let a: [u8; 256] = random_bytes();
                let b: [u8; 256] = random_bytes();
                group.bench_with_input(
                    BenchmarkId::new("array::threshold", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::array::threshold(
                                black_box(&a),
                                black_box(&b),
                                black_box(max),
                            ))
                        })
                    },
                );
            }
            _ => {}
        }

        // slice::threshold
        {
            let a = random_bytes_vec(size);
            let b = random_bytes_vec(size);
            group.bench_with_input(
                BenchmarkId::new("slice::threshold", format!("{}b", size * 8)),
                &size,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(hamming_bitwise_fast::slice::threshold(
                            black_box(&a),
                            black_box(&b),
                            black_box(max),
                        ))
                    })
                },
            );
        }

        // For comparison: full distance (no early exit)
        match size {
            128 => {
                let a: [u8; 128] = random_bytes();
                let b: [u8; 128] = random_bytes();
                group.bench_with_input(
                    BenchmarkId::new("array::distance (no early exit)", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::array::distance(
                                black_box(&a),
                                black_box(&b),
                            ))
                        })
                    },
                );
            }
            _ => {}
        }
    }

    group.finish();
}

// ============================================================================
// Batch threshold benchmarks (1000 comparisons with early exit)
// ============================================================================

fn batch_threshold_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_threshold");

    for &size in SIZES {
        let max = (size * 8) as u32 / 10; // ~10% threshold
        group.throughput(Throughput::Bits((size * 8 * 2 * BATCH) as u64));

        // array::batch_threshold - requires const generics
        match size {
            64 => {
                let source: [u8; 64] = random_bytes();
                let targets: Vec<[u8; 64]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];
                group.bench_with_input(
                    BenchmarkId::new("array::batch_threshold", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::array::batch_threshold(
                                black_box(&source),
                                black_box(&targets),
                                black_box(max),
                                black_box(&mut out),
                            ))
                        })
                    },
                );
            }
            96 => {
                let source: [u8; 96] = random_bytes();
                let targets: Vec<[u8; 96]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];
                group.bench_with_input(
                    BenchmarkId::new("array::batch_threshold", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::array::batch_threshold(
                                black_box(&source),
                                black_box(&targets),
                                black_box(max),
                                black_box(&mut out),
                            ))
                        })
                    },
                );
            }
            128 => {
                let source: [u8; 128] = random_bytes();
                let targets: Vec<[u8; 128]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];
                group.bench_with_input(
                    BenchmarkId::new("array::batch_threshold", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::array::batch_threshold(
                                black_box(&source),
                                black_box(&targets),
                                black_box(max),
                                black_box(&mut out),
                            ))
                        })
                    },
                );
            }
            256 => {
                let source: [u8; 256] = random_bytes();
                let targets: Vec<[u8; 256]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];
                group.bench_with_input(
                    BenchmarkId::new("array::batch_threshold", format!("{}b", size * 8)),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::array::batch_threshold(
                                black_box(&source),
                                black_box(&targets),
                                black_box(max),
                                black_box(&mut out),
                            ))
                        })
                    },
                );
            }
            _ => {}
        }

        // slice::batch_threshold
        {
            let source = random_bytes_vec(size);
            let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(size)).collect();
            let targets_refs: Vec<&[u8]> = targets.iter().map(|v| v.as_slice()).collect();
            let mut out = vec![0u32; BATCH];

            group.bench_with_input(
                BenchmarkId::new("slice::batch_threshold", format!("{}b", size * 8)),
                &size,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(hamming_bitwise_fast::slice::batch_threshold(
                            black_box(&source),
                            black_box(&targets_refs),
                            black_box(max),
                            black_box(&mut out),
                        ))
                    })
                },
            );
        }

        // For comparison: array::batch (no early exit) at 1024b
        match size {
            128 => {
                let source: [u8; 128] = random_bytes();
                let targets: Vec<[u8; 128]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];
                group.bench_with_input(
                    BenchmarkId::new("array::batch (no early exit)", format!("{}b", size * 8)),
                    &size,
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
            }
            _ => {}
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    single_benchmarks,
    batch_benchmarks,
    threshold_benchmarks,
    batch_threshold_benchmarks
);
criterion_main!(benches);

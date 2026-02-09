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
use helpers::{random_bytes, random_bytes_array};

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
// Helper: bench all single-comparison competitors for a given size
// ============================================================================

/// Bench all slice-based single competitors with shared data.
fn bench_single_slice(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    size: usize,
    a_vec: &[u8],
    b_vec: &[u8],
) {
    let label = format!("{}b", size * 8);

    group.bench_with_input(
        BenchmarkId::new("slice::distance", &label),
        &size,
        |bencher, _| {
            bencher.iter(|| {
                black_box(hamming_bitwise_fast::slice::distance(
                    black_box(a_vec),
                    black_box(b_vec),
                ))
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("hamming_bitwise_fast_v1", &label),
        &size,
        |bencher, _| bencher.iter(|| black_box(hamming_v1(black_box(a_vec), black_box(b_vec)))),
    );

    // simsimd — returns a normalized f64 (distance / total_bits), so it
    // performs additional floating-point division vs raw integer implementations.
    group.bench_with_input(
        BenchmarkId::new("simsimd", &label),
        &size,
        |bencher, _| {
            bencher.iter(|| {
                black_box(simsimd::BinarySimilarity::hamming(
                    black_box(a_vec),
                    black_box(b_vec),
                ))
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("hamming_crate", &label),
        &size,
        |bencher, _| {
            bencher
                .iter(|| black_box(hamming::distance_fast(black_box(a_vec), black_box(b_vec))))
        },
    );

    group.bench_with_input(
        BenchmarkId::new("triple_accel", &label),
        &size,
        |bencher, _| {
            bencher.iter(|| black_box(triple_accel::hamming(black_box(a_vec), black_box(b_vec))))
        },
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    group.bench_with_input(
        BenchmarkId::new("hamming_rs", &label),
        &size,
        |bencher, _| {
            bencher
                .iter(|| black_box(hamming_rs::distance_faster(black_box(a_vec), black_box(b_vec))))
        },
    );
}

// ============================================================================
// Single comparison benchmarks
// ============================================================================

fn single_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("single");

    for &size in SIZES {
        group.throughput(Throughput::Elements(1));

        match size {
            64 => {
                let a: [u8; 64] = random_bytes();
                let b: [u8; 64] = random_bytes();
                let a_vec = a.to_vec();
                let b_vec = b.to_vec();
                let label = format!("{}b", size * 8);

                group.bench_with_input(
                    BenchmarkId::new("array::distance", &label),
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

                bench_single_slice(&mut group, size, &a_vec, &b_vec);
            }
            96 => {
                let a: [u8; 96] = random_bytes();
                let b: [u8; 96] = random_bytes();
                let a_vec = a.to_vec();
                let b_vec = b.to_vec();
                let label = format!("{}b", size * 8);

                group.bench_with_input(
                    BenchmarkId::new("array::distance", &label),
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

                bench_single_slice(&mut group, size, &a_vec, &b_vec);
            }
            128 => {
                let a: [u8; 128] = random_bytes();
                let b: [u8; 128] = random_bytes();
                let a_vec = a.to_vec();
                let b_vec = b.to_vec();
                let label = format!("{}b", size * 8);

                group.bench_with_input(
                    BenchmarkId::new("array::distance", &label),
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

                bench_single_slice(&mut group, size, &a_vec, &b_vec);
            }
            256 => {
                let a: [u8; 256] = random_bytes();
                let b: [u8; 256] = random_bytes();
                let a_vec = a.to_vec();
                let b_vec = b.to_vec();
                let label = format!("{}b", size * 8);

                group.bench_with_input(
                    BenchmarkId::new("array::distance", &label),
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

                bench_single_slice(&mut group, size, &a_vec, &b_vec);
            }
            _ => {}
        }
    }

    group.finish();
}

// ============================================================================
// Helper: bench all batch competitors for a given size (slice-based)
// ============================================================================

fn bench_batch_slice(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    size: usize,
    source_vec: &[u8],
    targets_vecs: &[Vec<u8>],
) {
    let label = format!("{}b", size * 8);
    let targets_refs: Vec<&[u8]> = targets_vecs.iter().map(|v| v.as_slice()).collect();
    let mut out_u32 = vec![0u32; BATCH];

    group.bench_with_input(
        BenchmarkId::new("slice::batch", &label),
        &size,
        |bencher, _| {
            bencher.iter(|| {
                hamming_bitwise_fast::slice::batch(
                    black_box(source_vec),
                    black_box(&targets_refs),
                    black_box(&mut out_u32),
                );
                black_box(out_u32[0])
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("slice::distance (loop)", &label),
        &size,
        |bencher, _| {
            bencher.iter(|| {
                for (i, target) in black_box(targets_vecs).iter().enumerate() {
                    out_u32[i] =
                        hamming_bitwise_fast::slice::distance(black_box(source_vec), target);
                }
                black_box(out_u32[0])
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("hamming_bitwise_fast_v1", &label),
        &size,
        |bencher, _| {
            bencher.iter(|| {
                for (i, target) in black_box(targets_vecs).iter().enumerate() {
                    out_u32[i] = hamming_v1(black_box(source_vec), target);
                }
                black_box(out_u32[0])
            })
        },
    );

    // simsimd batch — returns a normalized f64 (distance / total_bits), so it
    // performs additional floating-point division vs raw integer implementations.
    {
        let mut out_f64 = vec![0f64; BATCH];
        group.bench_with_input(
            BenchmarkId::new("simsimd", &label),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    for (i, target) in black_box(targets_vecs).iter().enumerate() {
                        out_f64[i] =
                            simsimd::BinarySimilarity::hamming(black_box(source_vec), target)
                                .unwrap_or(0.0);
                    }
                    black_box(out_f64[0] as u64)
                })
            },
        );
    }

    {
        let mut out_u64 = vec![0u64; BATCH];
        group.bench_with_input(
            BenchmarkId::new("hamming_crate", &label),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    for (i, target) in black_box(targets_vecs).iter().enumerate() {
                        out_u64[i] =
                            hamming::distance_fast(black_box(source_vec), target).unwrap_or(0);
                    }
                    black_box(out_u64[0])
                })
            },
        );
    }

    group.bench_with_input(
        BenchmarkId::new("triple_accel", &label),
        &size,
        |bencher, _| {
            bencher.iter(|| {
                for (i, target) in black_box(targets_vecs).iter().enumerate() {
                    out_u32[i] = triple_accel::hamming(black_box(source_vec), target);
                }
                black_box(out_u32[0])
            })
        },
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let mut out_u64 = vec![0u64; BATCH];
        group.bench_with_input(
            BenchmarkId::new("hamming_rs", &label),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    for (i, target) in black_box(targets_vecs).iter().enumerate() {
                        out_u64[i] =
                            hamming_rs::distance_faster(black_box(source_vec), target);
                    }
                    black_box(out_u64[0])
                })
            },
        );
    }
}

// ============================================================================
// Batch comparison benchmarks (1000 comparisons)
// ============================================================================

fn batch_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch");

    for &size in SIZES {
        group.throughput(Throughput::Elements(BATCH as u64));

        match size {
            64 => {
                let source: [u8; 64] = random_bytes();
                let targets: Vec<[u8; 64]> = random_bytes_array(BATCH);
                let source_vec = source.to_vec();
                let targets_vecs: Vec<Vec<u8>> =
                    targets.iter().map(|t| t.to_vec()).collect();
                let mut out = vec![0u32; BATCH];
                let label = format!("{}b", size * 8);

                group.bench_with_input(
                    BenchmarkId::new("array::batch", &label),
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

                group.bench_with_input(
                    BenchmarkId::new("array::distance (loop)", &label),
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

                bench_batch_slice(&mut group, size, &source_vec, &targets_vecs);
            }
            96 => {
                let source: [u8; 96] = random_bytes();
                let targets: Vec<[u8; 96]> = random_bytes_array(BATCH);
                let source_vec = source.to_vec();
                let targets_vecs: Vec<Vec<u8>> =
                    targets.iter().map(|t| t.to_vec()).collect();
                let mut out = vec![0u32; BATCH];
                let label = format!("{}b", size * 8);

                group.bench_with_input(
                    BenchmarkId::new("array::batch", &label),
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

                group.bench_with_input(
                    BenchmarkId::new("array::distance (loop)", &label),
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

                bench_batch_slice(&mut group, size, &source_vec, &targets_vecs);
            }
            128 => {
                let source: [u8; 128] = random_bytes();
                let targets: Vec<[u8; 128]> = random_bytes_array(BATCH);
                let source_vec = source.to_vec();
                let targets_vecs: Vec<Vec<u8>> =
                    targets.iter().map(|t| t.to_vec()).collect();
                let mut out = vec![0u32; BATCH];
                let label = format!("{}b", size * 8);

                group.bench_with_input(
                    BenchmarkId::new("array::batch", &label),
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

                group.bench_with_input(
                    BenchmarkId::new("array::distance (loop)", &label),
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

                bench_batch_slice(&mut group, size, &source_vec, &targets_vecs);
            }
            256 => {
                let source: [u8; 256] = random_bytes();
                let targets: Vec<[u8; 256]> = random_bytes_array(BATCH);
                let source_vec = source.to_vec();
                let targets_vecs: Vec<Vec<u8>> =
                    targets.iter().map(|t| t.to_vec()).collect();
                let mut out = vec![0u32; BATCH];
                let label = format!("{}b", size * 8);

                group.bench_with_input(
                    BenchmarkId::new("array::batch", &label),
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

                group.bench_with_input(
                    BenchmarkId::new("array::distance (loop)", &label),
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

                bench_batch_slice(&mut group, size, &source_vec, &targets_vecs);
            }
            _ => {}
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
        group.throughput(Throughput::Elements(1));

        match size {
            64 => {
                let a: [u8; 64] = random_bytes();
                let b: [u8; 64] = random_bytes();
                let a_vec = a.to_vec();
                let b_vec = b.to_vec();
                let label = format!("{}b", size * 8);

                group.bench_with_input(
                    BenchmarkId::new("array::threshold", &label),
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

                group.bench_with_input(
                    BenchmarkId::new("slice::threshold", &label),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::slice::threshold(
                                black_box(&a_vec),
                                black_box(&b_vec),
                                black_box(max),
                            ))
                        })
                    },
                );
            }
            96 => {
                let a: [u8; 96] = random_bytes();
                let b: [u8; 96] = random_bytes();
                let a_vec = a.to_vec();
                let b_vec = b.to_vec();
                let label = format!("{}b", size * 8);

                group.bench_with_input(
                    BenchmarkId::new("array::threshold", &label),
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

                group.bench_with_input(
                    BenchmarkId::new("slice::threshold", &label),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::slice::threshold(
                                black_box(&a_vec),
                                black_box(&b_vec),
                                black_box(max),
                            ))
                        })
                    },
                );
            }
            128 => {
                let a: [u8; 128] = random_bytes();
                let b: [u8; 128] = random_bytes();
                let a_vec = a.to_vec();
                let b_vec = b.to_vec();
                let label = format!("{}b", size * 8);

                group.bench_with_input(
                    BenchmarkId::new("array::threshold", &label),
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

                group.bench_with_input(
                    BenchmarkId::new("slice::threshold", &label),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::slice::threshold(
                                black_box(&a_vec),
                                black_box(&b_vec),
                                black_box(max),
                            ))
                        })
                    },
                );

                // For comparison: full distance (no early exit)
                group.bench_with_input(
                    BenchmarkId::new("array::distance (no early exit)", &label),
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
                let a_vec = a.to_vec();
                let b_vec = b.to_vec();
                let label = format!("{}b", size * 8);

                group.bench_with_input(
                    BenchmarkId::new("array::threshold", &label),
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

                group.bench_with_input(
                    BenchmarkId::new("slice::threshold", &label),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::slice::threshold(
                                black_box(&a_vec),
                                black_box(&b_vec),
                                black_box(max),
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
        group.throughput(Throughput::Elements(BATCH as u64));

        match size {
            64 => {
                let source: [u8; 64] = random_bytes();
                let targets: Vec<[u8; 64]> = random_bytes_array(BATCH);
                let source_vec = source.to_vec();
                let targets_vecs: Vec<Vec<u8>> =
                    targets.iter().map(|t| t.to_vec()).collect();
                let targets_refs: Vec<&[u8]> =
                    targets_vecs.iter().map(|v| v.as_slice()).collect();
                let mut out = vec![0u32; BATCH];
                let label = format!("{}b", size * 8);

                group.bench_with_input(
                    BenchmarkId::new("array::batch_threshold", &label),
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

                group.bench_with_input(
                    BenchmarkId::new("slice::batch_threshold", &label),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::slice::batch_threshold(
                                black_box(&source_vec),
                                black_box(&targets_refs),
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
                let source_vec = source.to_vec();
                let targets_vecs: Vec<Vec<u8>> =
                    targets.iter().map(|t| t.to_vec()).collect();
                let targets_refs: Vec<&[u8]> =
                    targets_vecs.iter().map(|v| v.as_slice()).collect();
                let mut out = vec![0u32; BATCH];
                let label = format!("{}b", size * 8);

                group.bench_with_input(
                    BenchmarkId::new("array::batch_threshold", &label),
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

                group.bench_with_input(
                    BenchmarkId::new("slice::batch_threshold", &label),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::slice::batch_threshold(
                                black_box(&source_vec),
                                black_box(&targets_refs),
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
                let source_vec = source.to_vec();
                let targets_vecs: Vec<Vec<u8>> =
                    targets.iter().map(|t| t.to_vec()).collect();
                let targets_refs: Vec<&[u8]> =
                    targets_vecs.iter().map(|v| v.as_slice()).collect();
                let mut out = vec![0u32; BATCH];
                let label = format!("{}b", size * 8);

                group.bench_with_input(
                    BenchmarkId::new("array::batch_threshold", &label),
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

                group.bench_with_input(
                    BenchmarkId::new("slice::batch_threshold", &label),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::slice::batch_threshold(
                                black_box(&source_vec),
                                black_box(&targets_refs),
                                black_box(max),
                                black_box(&mut out),
                            ))
                        })
                    },
                );

                // For comparison: array::batch (no early exit) at 1024b
                group.bench_with_input(
                    BenchmarkId::new("array::batch (no early exit)", &label),
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
                let source_vec = source.to_vec();
                let targets_vecs: Vec<Vec<u8>> =
                    targets.iter().map(|t| t.to_vec()).collect();
                let targets_refs: Vec<&[u8]> =
                    targets_vecs.iter().map(|v| v.as_slice()).collect();
                let mut out = vec![0u32; BATCH];
                let label = format!("{}b", size * 8);

                group.bench_with_input(
                    BenchmarkId::new("array::batch_threshold", &label),
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

                group.bench_with_input(
                    BenchmarkId::new("slice::batch_threshold", &label),
                    &size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(hamming_bitwise_fast::slice::batch_threshold(
                                black_box(&source_vec),
                                black_box(&targets_refs),
                                black_box(max),
                                black_box(&mut out),
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

criterion_group!(
    benches,
    single_benchmarks,
    batch_benchmarks,
    threshold_benchmarks,
    batch_threshold_benchmarks
);
criterion_main!(benches);

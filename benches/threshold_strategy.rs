//! Benchmark: threshold strategies for batch_threshold.
//!
//! Compares the current chunked early-exit approach against a "compute full
//! distance then compare" approach. At small N, the full-compute approach
//! avoids chunking overhead and lets the compiler fully unroll the loop.
//!
//! Run with: cargo criterion --features multiversion_x86 --bench threshold_strategy

mod helpers;

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use helpers::{random_bytes, random_bytes_array};

const BATCH: usize = 1000;

// ============================================================================
// Strategy 1: Current approach (chunked early exit via threshold_impl)
// ============================================================================

// Uses the public API directly — this is the current implementation.

// ============================================================================
// Strategy 2: Full compute then compare
// ============================================================================

/// Batch threshold using full distance computation + comparison.
/// No early exit — computes full distance for every target, then checks.
#[cfg_attr(
    all(feature = "multiversion_x86", any(target_arch = "x86", target_arch = "x86_64")),
    multiversion::multiversion(targets(
        "x86_64+avx512vpopcntdq+avx512vl+popcnt",
        "x86_64+avx512bw+avx512vl+popcnt",
        "x86_64+avx2+popcnt",
        "x86_64+sse4.2+popcnt",
        "x86+avx2+popcnt",
        "x86+sse4.2+popcnt",
    ))
)]
#[inline]
fn batch_threshold_full_compute<const N: usize>(
    source: &[u8; N],
    targets: &[[u8; N]],
    max: u32,
    out: &mut [u32],
) -> u32 {
    assert_eq!(targets.len(), out.len());
    let mut best = u32::MAX;
    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let target = unsafe { &*opaque_ptr(target as *const [u8; N]) };
        let d = distance_impl(source, target);
        if d <= max {
            *dist = d;
            if d < best {
                best = d;
            }
        } else {
            *dist = u32::MAX;
        }
    }
    best
}

// ============================================================================
// Strategy 3: Hybrid — full compute for small arrays, chunked for large
// ============================================================================

/// Threshold block size (matches lib.rs).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const THRESHOLD_BLOCK_SIZE: usize = 64;
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
const THRESHOLD_BLOCK_SIZE: usize = 32;

/// If the array fits in 2 blocks or fewer, skip early exit entirely.
const FULL_COMPUTE_CUTOFF: usize = THRESHOLD_BLOCK_SIZE * 2;

#[cfg_attr(
    all(feature = "multiversion_x86", any(target_arch = "x86", target_arch = "x86_64")),
    multiversion::multiversion(targets(
        "x86_64+avx512vpopcntdq+avx512vl+popcnt",
        "x86_64+avx512bw+avx512vl+popcnt",
        "x86_64+avx2+popcnt",
        "x86_64+sse4.2+popcnt",
        "x86+avx2+popcnt",
        "x86+sse4.2+popcnt",
    ))
)]
#[inline]
fn batch_threshold_hybrid<const N: usize>(
    source: &[u8; N],
    targets: &[[u8; N]],
    max: u32,
    out: &mut [u32],
) -> u32 {
    assert_eq!(targets.len(), out.len());
    let mut best = u32::MAX;
    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let target = unsafe { &*opaque_ptr(target as *const [u8; N]) };

        // For small arrays: compute full distance (avoids chunking overhead)
        // For large arrays: use chunked early exit
        let result = if N <= FULL_COMPUTE_CUTOFF {
            let d = distance_impl(source, target);
            if d <= max { Some(d) } else { None }
        } else {
            threshold_impl(source, target, max)
        };

        match result {
            Some(d) => {
                *dist = d;
                if d < best {
                    best = d;
                }
            }
            None => {
                *dist = u32::MAX;
            }
        }
    }
    best
}

// ============================================================================
// Shared helpers (copied from lib.rs to avoid visibility issues)
// ============================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn opaque_ptr<T>(mut ptr: *const T) -> *const T {
    core::arch::asm!("/* {0} */", inout(reg) ptr, options(nomem, nostack, preserves_flags));
    ptr
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn distance_impl(a: &[u8], b: &[u8]) -> u32 {
    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);

    let main: u32 = a_chunks
        .clone()
        .zip(b_chunks.clone())
        .map(|(a, b)| {
            let a = u64::from_ne_bytes(a.try_into().unwrap());
            let b = u64::from_ne_bytes(b.try_into().unwrap());
            (a ^ b).count_ones()
        })
        .sum();

    let rem: u32 = a_chunks
        .remainder()
        .iter()
        .zip(b_chunks.remainder())
        .map(|(a, b)| (a ^ b).count_ones())
        .sum();

    main + rem
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline(always)]
fn distance_impl(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

#[inline(always)]
fn threshold_impl(a: &[u8], b: &[u8], threshold: u32) -> Option<u32> {
    let mut distance: u32 = 0;

    let a_blocks = a.chunks_exact(THRESHOLD_BLOCK_SIZE);
    let b_blocks = b.chunks_exact(THRESHOLD_BLOCK_SIZE);
    let a_rem = a_blocks.remainder();
    let b_rem = b_blocks.remainder();

    for (a_block, b_block) in a_blocks.zip(b_blocks) {
        distance += distance_impl(a_block, b_block);
        if distance > threshold {
            return None;
        }
    }

    distance += distance_impl(a_rem, b_rem);

    if distance <= threshold {
        Some(distance)
    } else {
        None
    }
}

/// Batch threshold using chunked early exit (same as the former public API).
#[cfg_attr(
    all(feature = "multiversion_x86", any(target_arch = "x86", target_arch = "x86_64")),
    multiversion::multiversion(targets(
        "x86_64+avx512vpopcntdq+avx512vl+popcnt",
        "x86_64+avx512bw+avx512vl+popcnt",
        "x86_64+avx2+popcnt",
        "x86_64+sse4.2+popcnt",
        "x86+avx2+popcnt",
        "x86+sse4.2+popcnt",
    ))
)]
#[inline]
fn batch_threshold_chunked<const N: usize>(
    source: &[u8; N],
    targets: &[[u8; N]],
    max: u32,
    out: &mut [u32],
) -> u32 {
    assert_eq!(targets.len(), out.len());
    let mut best = u32::MAX;
    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let target = unsafe { &*opaque_ptr(target as *const [u8; N]) };
        match threshold_impl(source, target, max) {
            Some(d) => {
                *dist = d;
                if d < best {
                    best = d;
                }
            }
            None => {
                *dist = u32::MAX;
            }
        }
    }
    best
}

// ============================================================================
// Benchmarks
// ============================================================================

fn threshold_strategy_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("threshold_strategy");

    macro_rules! bench_size {
        ($size:expr) => {{
            let max = ($size * 8) as u32 / 10; // ~10% threshold
            let label = format!("{}b", $size * 8);
            group.throughput(Throughput::Elements(BATCH as u64));

            let source: [u8; $size] = random_bytes();
            let targets: Vec<[u8; $size]> = random_bytes_array(BATCH);
            let mut out = vec![0u32; BATCH];

            // Current: chunked early exit (local implementation)
            group.bench_with_input(
                BenchmarkId::new("current_chunked", &label),
                &$size,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(batch_threshold_chunked(
                            black_box(&source),
                            black_box(&targets),
                            black_box(max),
                            black_box(&mut out),
                        ))
                    })
                },
            );

            // Full compute then compare
            group.bench_with_input(
                BenchmarkId::new("full_compute", &label),
                &$size,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(batch_threshold_full_compute(
                            black_box(&source),
                            black_box(&targets),
                            black_box(max),
                            black_box(&mut out),
                        ))
                    })
                },
            );

            // Hybrid
            group.bench_with_input(
                BenchmarkId::new("hybrid", &label),
                &$size,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(batch_threshold_hybrid(
                            black_box(&source),
                            black_box(&targets),
                            black_box(max),
                            black_box(&mut out),
                        ))
                    })
                },
            );

            // Plain batch (no threshold) for reference
            group.bench_with_input(
                BenchmarkId::new("plain_batch_ref", &label),
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
        }};
    }

    bench_size!(64);  // 512b
    bench_size!(96);  // 768b
    bench_size!(128); // 1024b
    bench_size!(160); // 1280b (2.5 blocks)
    bench_size!(192); // 1536b (3 blocks)
    bench_size!(256); // 2048b (4 blocks)
    bench_size!(512); // 4096b (8 blocks)

    group.finish();
}

criterion_group!(benches, threshold_strategy_benchmarks);
criterion_main!(benches);

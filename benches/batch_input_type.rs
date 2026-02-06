//! Benchmarks comparing array batch vs slice batch performance.
//!
//! # Background: The AVX-512 Gather Problem (Now Fixed)
//!
//! This benchmark was created to investigate a counterintuitive result where
//! `hamming_bitwise_slice_batch` was ~2-3x faster than `hamming_bitwise_array_batch`.
//!
//! ## Root Cause
//!
//! When LLVM sees `targets: &[[u8; N]]` (contiguous array of arrays) with
//! multiversion enabled, it tries to "optimize" by processing multiple targets
//! in parallel using AVX-512 gather instructions (VPGATHERQQ).
//!
//! Gather instructions are notoriously slow:
//! - Each element requires a separate memory fetch
//! - Cache line locality is destroyed
//! - Memory controller can't prefetch effectively
//! - Throughput is ~10-20x worse than contiguous loads
//!
//! ## The Fix
//!
//! We use `std::hint::black_box` on the target reference to prevent LLVM from
//! seeing across loop iterations and generating gather instructions. This forces
//! one-at-a-time processing with fast contiguous loads (VMOVDQU64).
//!
//! ## Assembly After Fix
//!
//! Both array_batch and slice_batch now generate optimal code:
//! ```asm
//! vmovdqu64 (%rsi),%zmm0            # Load 64 bytes contiguously
//! vmovdqu64 0x40(%rsi),%zmm1        # Next 64 bytes
//! vpopcntq %zmm0,%zmm0              # Hardware popcount
//! vpopcntq %zmm1,%zmm1
//! ```
//!
//! Run with: cargo bench --features multiversion_x86 --bench batch_input_type -- --quick

mod helpers;

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use hamming_bitwise_fast::{array, slice};
use helpers::{random_bytes, random_bytes_array, random_bytes_vec};

const BATCH: usize = 1000;

// ============================================================================
// Main benchmarks: comparing batch APIs
// ============================================================================

fn benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_input_type");

    macro_rules! bench_size {
        ($size:expr) => {{
            let bits = format!("{}b", $size * 8);

            // Array batch
            {
                let source: [u8; $size] = random_bytes();
                let targets: Vec<[u8; $size]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];

                group.bench_with_input(BenchmarkId::new("array_batch", &bits), &$size, |bencher, _| {
                    bencher.iter(|| {
                        array::batch(black_box(&source), black_box(&targets), &mut out);
                        black_box(out[0])
                    })
                });

                // Array via slice batch (convert arrays to slices)
                let targets_refs: Vec<&[u8]> = targets.iter().map(|a| a.as_slice()).collect();
                group.bench_with_input(BenchmarkId::new("array_via_slice_batch", &bits), &$size, |bencher, _| {
                    bencher.iter(|| {
                        slice::batch(
                            black_box(&source[..]),
                            black_box(&targets_refs),
                            &mut out,
                        );
                        black_box(out[0])
                    })
                });

                // Array loop (call single function repeatedly)
                group.bench_with_input(BenchmarkId::new("array_loop_single", &bits), &$size, |bencher, _| {
                    bencher.iter(|| {
                        for (target, dist) in black_box(&targets).iter().zip(out.iter_mut()) {
                            *dist = array::distance(black_box(&source), target);
                        }
                        black_box(out[0])
                    })
                });
            }

            // Slice batch
            {
                let source = random_bytes_vec($size);
                let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec($size)).collect();
                let targets_refs: Vec<&[u8]> = targets.iter().map(|v| v.as_slice()).collect();
                let mut out = vec![0u32; BATCH];

                group.bench_with_input(BenchmarkId::new("slice_batch", &bits), &$size, |bencher, _| {
                    bencher.iter(|| {
                        slice::batch(black_box(&source), black_box(&targets_refs), &mut out);
                        black_box(out[0])
                    })
                });

                // Slice loop (call single function repeatedly)
                group.bench_with_input(BenchmarkId::new("slice_loop_single", &bits), &$size, |bencher, _| {
                    bencher.iter(|| {
                        for (target, dist) in black_box(&targets).iter().zip(out.iter_mut()) {
                            *dist = slice::distance(black_box(&source), target);
                        }
                        black_box(out[0])
                    })
                });
            }
        }};
    }

    bench_size!(64);
    bench_size!(128);
    bench_size!(256);

    group.finish();
}

// ============================================================================
// Demonstration: black_box prevents slow gather instructions on x86
// (Only meaningful with --features multiversion_x86)
// ============================================================================

#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
mod gather_demo {
    use super::*;

    /// Inner function WITHOUT black_box - LLVM generates slow gather instructions.
    #[multiversion::multiversion(targets(
        "x86_64+avx512vpopcntdq+avx512vl+popcnt",
        "x86_64+avx512bw+avx512vl+popcnt",
        "x86_64+avx2+popcnt",
        "x86_64+sse4.2+popcnt",
    ))]
    fn batch_no_black_box<const N: usize>(source: &[u8; N], targets: &[[u8; N]], out: &mut [u32]) {
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            // No black_box - compiler can see across iterations and use gather
            let a_chunks = source.chunks_exact(8);
            let b_chunks = target.chunks_exact(8);

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

            *dist = main + rem;
        }
    }

    /// Inner function WITH black_box - prevents gather, uses fast contiguous loads.
    #[multiversion::multiversion(targets(
        "x86_64+avx512vpopcntdq+avx512vl+popcnt",
        "x86_64+avx512bw+avx512vl+popcnt",
        "x86_64+avx2+popcnt",
        "x86_64+sse4.2+popcnt",
    ))]
    fn batch_with_black_box<const N: usize>(source: &[u8; N], targets: &[[u8; N]], out: &mut [u32]) {
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            // black_box hides target from cross-iteration optimization
            let target: &[u8] = std::hint::black_box(target);

            let a_chunks = source.chunks_exact(8);
            let b_chunks = target.chunks_exact(8);

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

            *dist = main + rem;
        }
    }

    pub fn benchmarks(c: &mut Criterion) {
        let mut group = c.benchmark_group("gather_demo");

        macro_rules! bench_size {
            ($size:expr) => {{
                let bits = format!("{}b", $size * 8);
                let source: [u8; $size] = random_bytes();
                let targets: Vec<[u8; $size]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];

                group.bench_with_input(
                    BenchmarkId::new("no_blackbox_slow_gather", &bits),
                    &$size,
                    |bencher, _| {
                        bencher.iter(|| {
                            batch_no_black_box(black_box(&source), black_box(&targets), &mut out);
                            black_box(out[0])
                        })
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("blackbox_fast_loads", &bits),
                    &$size,
                    |bencher, _| {
                        bencher.iter(|| {
                            batch_with_black_box(black_box(&source), black_box(&targets), &mut out);
                            black_box(out[0])
                        })
                    },
                );
            }};
        }

        bench_size!(64);
        bench_size!(128);
        bench_size!(256);

        group.finish();
    }
}

#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
criterion_group!(benches, benchmarks, gather_demo::benchmarks);

#[cfg(not(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
)))]
criterion_group!(benches, benchmarks);

criterion_main!(benches);

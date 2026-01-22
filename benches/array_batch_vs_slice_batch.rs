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
//! Run with: cargo bench --features multiversion_x86 --bench array_batch_vs_slice_batch

mod helpers;

use hamming_bitwise_fast::{
    hamming_bitwise_array, hamming_bitwise_array_batch, hamming_bitwise_slice,
    hamming_bitwise_slice_batch,
};
use helpers::{random_bytes, random_bytes_array, random_bytes_vec};

fn main() {
    divan::main();
}

const BATCH: usize = 1000;

// ============================================================================
// Direct comparison: array batch vs slice batch
// ============================================================================

#[divan::bench(consts = [64, 128, 256])]
fn array_batch<const N: usize>(bencher: divan::Bencher) {
    let source: [u8; N] = random_bytes();
    let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
    let mut out = vec![0u32; BATCH];

    bencher.bench_local(|| {
        hamming_bitwise_array_batch(&source, &targets, &mut out);
        out[0]
    });
}

#[divan::bench(args = [64, 128, 256])]
fn slice_batch(bencher: divan::Bencher, bytes: usize) {
    let source = random_bytes_vec(bytes);
    let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(bytes)).collect();
    let targets_refs: Vec<&[u8]> = targets.iter().map(|v| v.as_slice()).collect();
    let mut out = vec![0u32; BATCH];

    bencher.bench_local(|| {
        hamming_bitwise_slice_batch(&source, &targets_refs, &mut out);
        out[0]
    });
}

// ============================================================================
// Workaround: Convert arrays to slices for better performance
// ============================================================================

/// Shows how to get best performance with array data: convert to slices first.
#[divan::bench(consts = [64, 128, 256])]
fn array_as_slice_batch<const N: usize>(bencher: divan::Bencher) {
    let source: [u8; N] = random_bytes();
    let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
    // Convert arrays to slices - this overhead is minimal
    let targets_refs: Vec<&[u8]> = targets.iter().map(|a| a.as_slice()).collect();
    let mut out = vec![0u32; BATCH];

    bencher.bench_local(|| {
        hamming_bitwise_slice_batch(&source[..], &targets_refs, &mut out);
        out[0]
    });
}

// ============================================================================
// Iterator-based: process one at a time (avoids gather optimization)
// ============================================================================

/// Array iterator: calls hamming_bitwise_array in a loop.
/// This should avoid gather instructions since each call is independent.
#[divan::bench(consts = [64, 128, 256])]
fn array_iter<const N: usize>(bencher: divan::Bencher) {
    let source: [u8; N] = random_bytes();
    let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
    let mut out = vec![0u32; BATCH];

    bencher.bench_local(|| {
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            *dist = hamming_bitwise_array(&source, target);
        }
        out[0]
    });
}

/// Slice iterator: calls hamming_bitwise_slice in a loop.
#[divan::bench(args = [64, 128, 256])]
fn slice_iter(bencher: divan::Bencher, bytes: usize) {
    let source = random_bytes_vec(bytes);
    let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(bytes)).collect();
    let mut out = vec![0u32; BATCH];

    bencher.bench_local(|| {
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            *dist = hamming_bitwise_slice(&source, target);
        }
        out[0]
    });
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

    /// Demonstrates the ~2-3x performance penalty from VPGATHERQQ.
    #[divan::bench(consts = [64, 128, 256])]
    pub fn without_black_box<const N: usize>(bencher: divan::Bencher) {
        let source: [u8; N] = random_bytes();
        let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            batch_no_black_box(&source, &targets, &mut out);
            out[0]
        });
    }

    /// Shows the fix: black_box prevents gather and restores fast performance.
    #[divan::bench(consts = [64, 128, 256])]
    pub fn with_black_box<const N: usize>(bencher: divan::Bencher) {
        let source: [u8; N] = random_bytes();
        let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            batch_with_black_box(&source, &targets, &mut out);
            out[0]
        });
    }
}

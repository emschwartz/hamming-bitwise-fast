//! Compares batch operations vs individual function calls.
//!
//! Key insight: Batch APIs allow the compiler to optimize the entire loop,
//! potentially enabling better instruction scheduling and cache utilization.
//!
//! Run with: cargo bench --bench batch_vs_single

mod helpers;

use helpers::{random_bytes, random_bytes_array, random_bytes_vec};

fn main() {
    divan::main();
}

const BATCH: usize = 64;

// ============================================================================
// Array implementations
// ============================================================================

/// Single array comparison using u64 chunks.
#[inline(always)]
fn hamming_array<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
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

/// Batch array comparison.
#[inline]
fn hamming_array_batch<const N: usize>(source: &[u8; N], targets: &[[u8; N]], out: &mut [u32]) {
    assert_eq!(targets.len(), out.len());
    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        *dist = hamming_array(source, target);
    }
}

// ============================================================================
// Slice implementations
// ============================================================================

/// Single slice comparison using u64 chunks.
#[inline(always)]
fn hamming_slice(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
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

/// Batch slice comparison.
#[inline]
fn hamming_slice_batch(source: &[u8], targets: &[&[u8]], out: &mut [u32]) {
    assert_eq!(targets.len(), out.len());
    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        *dist = hamming_slice(source, target);
    }
}

// ============================================================================
// Array benchmarks
// ============================================================================

mod array {
    use super::*;

    #[divan::bench(consts = [64, 96, 128, 256])]
    fn single_loop<const N: usize>(bencher: divan::Bencher) {
        let source: [u8; N] = random_bytes();
        let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            for (target, dist) in targets.iter().zip(out.iter_mut()) {
                *dist = hamming_array(&source, target);
            }
            out[0]
        });
    }

    #[divan::bench(consts = [64, 96, 128, 256])]
    fn batch<const N: usize>(bencher: divan::Bencher) {
        let source: [u8; N] = random_bytes();
        let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            hamming_array_batch(&source, &targets, &mut out);
            out[0]
        });
    }
}

// ============================================================================
// Slice benchmarks
// ============================================================================

mod slice {
    use super::*;

    #[divan::bench(args = [64, 96, 128, 256])]
    fn single_loop(bencher: divan::Bencher, bytes: usize) {
        let source = random_bytes_vec(bytes);
        let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(bytes)).collect();
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            for (target, dist) in targets.iter().zip(out.iter_mut()) {
                *dist = hamming_slice(&source, target);
            }
            out[0]
        });
    }

    #[divan::bench(args = [64, 96, 128, 256])]
    fn batch(bencher: divan::Bencher, bytes: usize) {
        let source = random_bytes_vec(bytes);
        let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(bytes)).collect();
        let targets_refs: Vec<&[u8]> = targets.iter().map(|v| v.as_slice()).collect();
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            hamming_slice_batch(&source, &targets_refs, &mut out);
            out[0]
        });
    }
}

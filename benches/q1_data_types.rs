//! Q1: What's the fastest way to compute Hamming distance on u8 arrays?
//!
//! Key questions:
//! - Is byte-by-byte iteration faster or slower than chunks_exact(8)?
//! - Does the remainder handling in chunks_exact add overhead?
//!
//! Run with: cargo bench --bench q1_data_types

mod helpers;

use helpers::random_bytes;

fn main() {
    divan::main();
}

// ============================================================================
// Implementations to benchmark
// ============================================================================

/// Byte-by-byte iteration.
#[inline]
fn hamming_u8_iter<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Process as u64 chunks (no remainder handling).
#[inline]
fn hamming_u8_chunks<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    a.chunks_exact(8)
        .zip(b.chunks_exact(8))
        .map(|(a_chunk, b_chunk)| {
            let a_val = u64::from_ne_bytes(a_chunk.try_into().unwrap());
            let b_val = u64::from_ne_bytes(b_chunk.try_into().unwrap());
            (a_val ^ b_val).count_ones()
        })
        .sum()
}

/// Process as u64 chunks with byte-by-byte remainder handling.
#[inline]
fn hamming_u8_chunks_with_remainder<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);

    let main: u32 = a_chunks
        .clone()
        .zip(b_chunks.clone())
        .map(|(a_chunk, b_chunk)| {
            let a_val = u64::from_ne_bytes(a_chunk.try_into().unwrap());
            let b_val = u64::from_ne_bytes(b_chunk.try_into().unwrap());
            (a_val ^ b_val).count_ones()
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

// ============================================================================
// Benchmarks
// ============================================================================

#[divan::bench(consts = [64, 96, 128, 256])]
fn u8_iter<const N: usize>(bencher: divan::Bencher) {
    let a: [u8; N] = random_bytes();
    let b: [u8; N] = random_bytes();
    bencher.bench_local(|| hamming_u8_iter(&a, &b));
}

#[divan::bench(consts = [64, 96, 128, 256])]
fn u8_chunks<const N: usize>(bencher: divan::Bencher) {
    let a: [u8; N] = random_bytes();
    let b: [u8; N] = random_bytes();
    bencher.bench_local(|| hamming_u8_chunks(&a, &b));
}

#[divan::bench(consts = [64, 96, 128, 256])]
fn u8_chunks_with_remainder<const N: usize>(bencher: divan::Bencher) {
    let a: [u8; N] = random_bytes();
    let b: [u8; N] = random_bytes();
    bencher.bench_local(|| hamming_u8_chunks_with_remainder(&a, &b));
}

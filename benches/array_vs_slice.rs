//! Compares fixed-size arrays vs dynamic slices.
//!
//! Key insight: Fixed-size arrays allow the compiler to unroll loops and
//! optimize bounds checks away, while slices require runtime length checks.
//!
//! Run with: cargo bench --bench array_vs_slice

mod helpers;

use helpers::{random_bytes, random_bytes_vec};

fn main() {
    divan::main();
}

// ============================================================================
// Implementations
// ============================================================================

/// Fixed-size array: u64 chunks with remainder.
#[inline]
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

/// Dynamic slice: u64 chunks with remainder.
#[inline]
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

// ============================================================================
// Benchmarks (sizes in bits: 512b, 768b, 1024b, 2048b = 64, 96, 128, 256 bytes)
// ============================================================================

#[divan::bench(consts = [64, 96, 128, 256])]
fn array<const N: usize>(bencher: divan::Bencher) {
    let a: [u8; N] = random_bytes();
    let b: [u8; N] = random_bytes();
    bencher.bench_local(|| hamming_array(&a, &b));
}

#[divan::bench(args = [64, 96, 128, 256])]
fn slice(bencher: divan::Bencher, bytes: usize) {
    let a = random_bytes_vec(bytes);
    let b = random_bytes_vec(bytes);
    bencher.bench_local(|| hamming_slice(&a, &b));
}

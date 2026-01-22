//! Q6: Does multiversion improve slice performance?
//!
//! Key questions:
//! - Does runtime CPU dispatch benefit slice operations?
//! - How does the u64 chunks approach compare to simple iteration?
//!
//! Run with: cargo bench --bench q6_slice_multiversion
//! Compare with: RUSTFLAGS="-C target-cpu=native" cargo bench --bench q6_slice_multiversion

mod helpers;

use helpers::{random_bytes, random_bytes_vec};

fn main() {
    divan::main();
}

// ============================================================================
// Slice implementations
// ============================================================================

/// Simple byte-by-byte iteration.
#[inline]
fn slice_iter(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// u64 chunks with remainder (what the library uses on x86).
#[inline]
fn slice_u64_chunks(a: &[u8], b: &[u8]) -> u32 {
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
// Array implementation for comparison
// ============================================================================

#[inline]
fn array_u64_chunks<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
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
// Benchmarks
// ============================================================================

#[divan::bench(args = [64, 96, 128, 256])]
fn slice_iter_bench(bencher: divan::Bencher, bytes: usize) {
    let a = random_bytes_vec(bytes);
    let b = random_bytes_vec(bytes);
    bencher.bench_local(|| slice_iter(&a, &b));
}

#[divan::bench(args = [64, 96, 128, 256])]
fn slice_u64_chunks_bench(bencher: divan::Bencher, bytes: usize) {
    let a = random_bytes_vec(bytes);
    let b = random_bytes_vec(bytes);
    bencher.bench_local(|| slice_u64_chunks(&a, &b));
}

#[divan::bench(consts = [64, 96, 128, 256])]
fn array_u64_chunks_bench<const N: usize>(bencher: divan::Bencher) {
    let a: [u8; N] = random_bytes();
    let b: [u8; N] = random_bytes();
    bencher.bench_local(|| array_u64_chunks(&a, &b));
}

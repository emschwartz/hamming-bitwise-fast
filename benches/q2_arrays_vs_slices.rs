//! Q2: Are arrays better than slices? How much does it matter?
//!
//! Key questions:
//! - Does the compiler optimize fixed-size arrays better?
//! - Can processing as u64 chunks negate the slice overhead?
//! - Does asserting the slice length is a multiple of 8 help?
//!
//! Run with: cargo bench --bench q2_arrays_vs_slices

mod helpers;

use helpers::{random_bytes, random_bytes_vec};

fn main() {
    divan::main();
}

// ============================================================================
// Array implementations
// ============================================================================

/// Array: byte-by-byte iteration.
#[inline]
fn array_iter<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Array: u64 chunks with remainder.
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
// Slice implementations
// ============================================================================

/// Slice: byte-by-byte iteration.
#[inline]
fn slice_iter(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Slice: u64 chunks with remainder.
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
// Benchmarks: Arrays
// ============================================================================

#[divan::bench(consts = [64, 96, 128, 256])]
fn array_iter_bench<const N: usize>(bencher: divan::Bencher) {
    let a: [u8; N] = random_bytes();
    let b: [u8; N] = random_bytes();
    bencher.bench_local(|| array_iter(&a, &b));
}

#[divan::bench(consts = [64, 96, 128, 256])]
fn array_u64_chunks_bench<const N: usize>(bencher: divan::Bencher) {
    let a: [u8; N] = random_bytes();
    let b: [u8; N] = random_bytes();
    bencher.bench_local(|| array_u64_chunks(&a, &b));
}

// ============================================================================
// Benchmarks: Slices
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

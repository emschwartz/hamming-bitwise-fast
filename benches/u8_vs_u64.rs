//! Compares byte-by-byte iteration (u8) vs u64 chunk processing.
//!
//! Key insight: On ARM, simple u8 iteration auto-vectorizes well with NEON.
//! On x86, u64 chunk processing enables AVX-512 VPOPCNTDQ when available.
//!
//! Run with: cargo bench --bench u8_vs_u64
//! Compare with: RUSTFLAGS="-C target-cpu=native" cargo bench --bench u8_vs_u64

mod helpers;

use helpers::random_bytes;

fn main() {
    divan::main();
}

// ============================================================================
// Implementations
// ============================================================================

/// Byte-by-byte iteration - simple, auto-vectorizes well on ARM.
#[inline]
fn hamming_u8_iter<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Process as u64 chunks - enables AVX-512 VPOPCNTDQ on x86.
#[inline]
fn hamming_u64_chunks<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
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
// Benchmarks (sizes in bits: 512b, 768b, 1024b, 2048b = 64, 96, 128, 256 bytes)
// ============================================================================

#[divan::bench(consts = [64, 96, 128, 256])]
fn u8_iter<const N: usize>(bencher: divan::Bencher) {
    let a: [u8; N] = random_bytes();
    let b: [u8; N] = random_bytes();
    bencher.bench_local(|| hamming_u8_iter(&a, &b));
}

#[divan::bench(consts = [64, 96, 128, 256])]
fn u64_chunks<const N: usize>(bencher: divan::Bencher) {
    let a: [u8; N] = random_bytes();
    let b: [u8; N] = random_bytes();
    bencher.bench_local(|| hamming_u64_chunks(&a, &b));
}

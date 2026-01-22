//! Q3: Which SIMD instructions are most beneficial? How to target them effectively?
//!
//! Key questions:
//! - On ARM: NEON is great, but is it used by default?
//! - On x86: vectorized POPCNT (AVX-512 VPOPCNT) is massive - how to enable it?
//! - Which dispatch strategy works best: multiversion or RUSTFLAGS?
//!
//! How to test different compiler optimizations:
//! ```sh
//! # Default (baseline)
//! cargo bench --bench q3_simd_dispatch
//!
//! # With target-cpu=native (uses all CPU features)
//! RUSTFLAGS="-C target-cpu=native" cargo bench --bench q3_simd_dispatch
//! ```
//!
//! Run with: cargo bench --bench q3_simd_dispatch

mod helpers;

use helpers::random_bytes;

fn main() {
    divan::main();
}

// ============================================================================
// Implementations to benchmark
// ============================================================================

/// Byte-by-byte iteration - simple, auto-vectorizes well on ARM.
#[inline]
fn hamming_iter<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
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

#[divan::bench(consts = [64, 96, 128, 256])]
fn iter<const N: usize>(bencher: divan::Bencher) {
    let a: [u8; N] = random_bytes();
    let b: [u8; N] = random_bytes();
    bencher.bench_local(|| hamming_iter(&a, &b));
}

#[divan::bench(consts = [64, 96, 128, 256])]
fn u64_chunks<const N: usize>(bencher: divan::Bencher) {
    let a: [u8; N] = random_bytes();
    let b: [u8; N] = random_bytes();
    bencher.bench_local(|| hamming_u64_chunks(&a, &b));
}

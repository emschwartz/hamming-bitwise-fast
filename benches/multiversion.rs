//! Compares static compilation vs runtime CPU dispatch (multiversion).
//!
//! Run this benchmark twice to see the full picture:
//!
//! ```sh
//! # Run 1: Default (SSE2) vs multiversion
//! cargo bench --bench multiversion --features multiversion_x86
//!
//! # Run 2: Native (all CPU features) vs multiversion
//! RUSTFLAGS="-C target-cpu=native" cargo bench --bench multiversion --features multiversion_x86
//! ```
//!
//! This answers:
//! - Run 1: How much does multiversion help over baseline?
//! - Run 2: Is there overhead from runtime dispatch vs compile-time targeting?
//!
//! Note: This benchmark only runs on x86/x86_64 with the multiversion_x86 feature.

#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
mod helpers;

#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
use helpers::random_bytes;

fn main() {
    #[cfg(all(
        feature = "multiversion_x86",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    {
        divan::main();
    }

    #[cfg(not(all(
        feature = "multiversion_x86",
        any(target_arch = "x86", target_arch = "x86_64")
    )))]
    {
        eprintln!("This benchmark requires x86/x86_64 with --features multiversion_x86");
        eprintln!("Run: cargo bench --bench multiversion --features multiversion_x86");
    }
}

// ============================================================================
// Static implementation (affected by RUSTFLAGS)
// ============================================================================

#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[inline(always)]
fn hamming_static<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
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
// Runtime dispatch implementation (multiversion)
// ============================================================================

#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[multiversion::multiversion(targets(
    "x86_64+avx512vpopcntdq+avx512vl+popcnt",
    "x86_64+avx512bw+avx512vl+popcnt",
    "x86_64+avx2+popcnt",
    "x86_64+sse4.2+popcnt",
    "x86+avx2+popcnt",
    "x86+sse4.2+popcnt",
))]
#[inline(always)]
fn hamming_multiversion<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
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

#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[divan::bench(consts = [64, 96, 128, 256])]
fn static_compile<const N: usize>(bencher: divan::Bencher) {
    let a: [u8; N] = random_bytes();
    let b: [u8; N] = random_bytes();
    bencher.bench_local(|| hamming_static(&a, &b));
}

#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[divan::bench(consts = [64, 96, 128, 256])]
fn multiversion<const N: usize>(bencher: divan::Bencher) {
    let a: [u8; N] = random_bytes();
    let b: [u8; N] = random_bytes();
    bencher.bench_local(|| hamming_multiversion(&a, &b));
}

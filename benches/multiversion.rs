//! Compares static compilation vs runtime CPU dispatch (multiversion).
//!
//! Run this benchmark twice to see the full picture:
//!
//! ```sh
//! # Run 1: Default (SSE2) vs multiversion
//! cargo bench --bench multiversion --features multiversion_x86 -- --quick
//!
//! # Run 2: Native (all CPU features) vs multiversion
//! RUSTFLAGS="-C target-cpu=native" cargo bench --bench multiversion --features multiversion_x86 -- --quick
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
mod benches {
    mod helpers;

    use std::hint::black_box;

    use criterion::{criterion_group, BenchmarkId, Criterion};
    use helpers::random_bytes;

    // ============================================================================
    // Static implementation (affected by RUSTFLAGS)
    // ============================================================================

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

    fn static_compile_benchmarks(c: &mut Criterion) {
        let mut group = c.benchmark_group("static_compile");

        macro_rules! bench_size {
            ($size:expr) => {{
                let a: [u8; $size] = random_bytes();
                let b: [u8; $size] = random_bytes();
                group.bench_with_input(
                    BenchmarkId::from_parameter($size),
                    &$size,
                    |bencher, _| {
                        bencher.iter(|| black_box(hamming_static(black_box(&a), black_box(&b))))
                    },
                );
            }};
        }

        bench_size!(64);
        bench_size!(96);
        bench_size!(128);
        bench_size!(256);

        group.finish();
    }

    fn multiversion_benchmarks(c: &mut Criterion) {
        let mut group = c.benchmark_group("multiversion");

        macro_rules! bench_size {
            ($size:expr) => {{
                let a: [u8; $size] = random_bytes();
                let b: [u8; $size] = random_bytes();
                group.bench_with_input(
                    BenchmarkId::from_parameter($size),
                    &$size,
                    |bencher, _| {
                        bencher
                            .iter(|| black_box(hamming_multiversion(black_box(&a), black_box(&b))))
                    },
                );
            }};
        }

        bench_size!(64);
        bench_size!(96);
        bench_size!(128);
        bench_size!(256);

        group.finish();
    }

    criterion_group!(
        multiversion_benches,
        static_compile_benchmarks,
        multiversion_benchmarks
    );
}

#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
fn main() {
    benches::multiversion_benches();
}

#[cfg(not(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
)))]
fn main() {
    eprintln!("This benchmark requires x86/x86_64 with --features multiversion_x86");
    eprintln!("Run: cargo bench --bench multiversion --features multiversion_x86");
}

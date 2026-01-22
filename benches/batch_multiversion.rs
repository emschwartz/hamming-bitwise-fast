//! Compares inlining strategies when using multiversion for batch operations.
//!
//! When using runtime CPU dispatch, where should `#[multiversion]` go?
//!
//! 1. **dispatch_per_call** - `#[multiversion]` on single function, called in loop
//!    - Each iteration pays dispatch cost (but may be optimized away)
//! 2. **dispatch_once_call** - `#[multiversion]` on batch, calls `#[inline(always)]` single
//!    - Dispatch once, function call per iteration
//! 3. **dispatch_once_inlined** - `#[multiversion]` on batch with body inlined
//!    - Dispatch once, no function call overhead
//!
//! Run with: cargo bench --bench batch_multiversion --features multiversion_x86 -- --quick
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
    use helpers::{random_bytes, random_bytes_array};

    const BATCH: usize = 64;

    // ============================================================================
    // Strategy 1: Dispatch per call
    // ============================================================================

    mod dispatch_per_call {
        #[multiversion::multiversion(targets(
            "x86_64+avx512vpopcntdq+avx512vl+popcnt",
            "x86_64+avx512bw+avx512vl+popcnt",
            "x86_64+avx2+popcnt",
            "x86_64+sse4.2+popcnt",
            "x86+avx2+popcnt",
            "x86+sse4.2+popcnt",
        ))]
        #[inline(always)]
        fn hamming_single<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
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

        #[inline]
        pub fn batch<const N: usize>(source: &[u8; N], targets: &[[u8; N]], out: &mut [u32]) {
            assert_eq!(targets.len(), out.len());
            for (target, dist) in targets.iter().zip(out.iter_mut()) {
                *dist = hamming_single(source, target);
            }
        }
    }

    // ============================================================================
    // Strategy 2: Dispatch once, call #[inline(always)] function in loop
    // ============================================================================

    mod dispatch_once_call {
        #[inline(always)]
        fn hamming_single<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
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

        #[multiversion::multiversion(targets(
            "x86_64+avx512vpopcntdq+avx512vl+popcnt",
            "x86_64+avx512bw+avx512vl+popcnt",
            "x86_64+avx2+popcnt",
            "x86_64+sse4.2+popcnt",
            "x86+avx2+popcnt",
            "x86+sse4.2+popcnt",
        ))]
        #[inline]
        pub fn batch<const N: usize>(source: &[u8; N], targets: &[[u8; N]], out: &mut [u32]) {
            assert_eq!(targets.len(), out.len());
            for (target, dist) in targets.iter().zip(out.iter_mut()) {
                *dist = hamming_single(source, target);
            }
        }
    }

    // ============================================================================
    // Strategy 3: Dispatch once, body fully inlined
    // ============================================================================

    mod dispatch_once_inlined {
        #[multiversion::multiversion(targets(
            "x86_64+avx512vpopcntdq+avx512vl+popcnt",
            "x86_64+avx512bw+avx512vl+popcnt",
            "x86_64+avx2+popcnt",
            "x86_64+sse4.2+popcnt",
            "x86+avx2+popcnt",
            "x86+sse4.2+popcnt",
        ))]
        #[inline]
        pub fn batch<const N: usize>(source: &[u8; N], targets: &[[u8; N]], out: &mut [u32]) {
            assert_eq!(targets.len(), out.len());

            for (target, dist) in targets.iter().zip(out.iter_mut()) {
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
    }

    // ============================================================================
    // Benchmarks
    // ============================================================================

    fn dispatch_per_call_benchmarks(c: &mut Criterion) {
        let mut group = c.benchmark_group("dispatch_per_call");

        macro_rules! bench_size {
            ($size:expr) => {{
                let source: [u8; $size] = random_bytes();
                let targets: Vec<[u8; $size]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];
                group.bench_with_input(
                    BenchmarkId::from_parameter($size),
                    &$size,
                    |bencher, _| {
                        bencher.iter(|| {
                            dispatch_per_call::batch(
                                black_box(&source),
                                black_box(&targets),
                                &mut out,
                            );
                            black_box(out[0])
                        })
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

    fn dispatch_once_call_benchmarks(c: &mut Criterion) {
        let mut group = c.benchmark_group("dispatch_once_call");

        macro_rules! bench_size {
            ($size:expr) => {{
                let source: [u8; $size] = random_bytes();
                let targets: Vec<[u8; $size]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];
                group.bench_with_input(
                    BenchmarkId::from_parameter($size),
                    &$size,
                    |bencher, _| {
                        bencher.iter(|| {
                            dispatch_once_call::batch(
                                black_box(&source),
                                black_box(&targets),
                                &mut out,
                            );
                            black_box(out[0])
                        })
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

    fn dispatch_once_inlined_benchmarks(c: &mut Criterion) {
        let mut group = c.benchmark_group("dispatch_once_inlined");

        macro_rules! bench_size {
            ($size:expr) => {{
                let source: [u8; $size] = random_bytes();
                let targets: Vec<[u8; $size]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];
                group.bench_with_input(
                    BenchmarkId::from_parameter($size),
                    &$size,
                    |bencher, _| {
                        bencher.iter(|| {
                            dispatch_once_inlined::batch(
                                black_box(&source),
                                black_box(&targets),
                                &mut out,
                            );
                            black_box(out[0])
                        })
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
        batch_multiversion_benches,
        dispatch_per_call_benchmarks,
        dispatch_once_call_benchmarks,
        dispatch_once_inlined_benchmarks
    );
}

#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
fn main() {
    benches::batch_multiversion_benches();
}

#[cfg(not(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
)))]
fn main() {
    eprintln!("This benchmark requires x86/x86_64 with --features multiversion_x86");
    eprintln!("Run: cargo bench --bench batch_multiversion --features multiversion_x86");
}

//! Compares multiversion placement strategies for batch operations.
//!
//! When using runtime CPU dispatch with `#[multiversion]`, where should the
//! annotation go? This benchmark tests various combinations:
//!
//! **For slices** (`&[&[u8]]` - non-contiguous pointers):
//! - Inlining the body is fastest (no gather concern)
//!
//! **For arrays** (`&[[u8; N]]` - contiguous memory):
//! - Calling MV single prevents slow VPGATHERQQ gather instructions
//! - Body inlining triggers gather optimization (slow!)
//!
//! Run with: cargo bench --features multiversion_x86 --bench multiversion_placement -- --quick

mod helpers;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

use helpers::{random_bytes, random_bytes_array, random_bytes_vec};

const BATCH: usize = 64;

// ============================================================================
// Private implementations WITHOUT multiversion
// ============================================================================

#[inline(always)]
fn slice_impl(a: &[u8], b: &[u8]) -> u32 {
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

#[inline(always)]
fn array_impl<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
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
// Multiversion configurations (x86 with feature)
// ============================================================================

#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
mod mv {
    use super::{array_impl, slice_impl};

    // ========================================================================
    // SLICE strategies
    // ========================================================================

    // MV batch calling #[inline(always)] impl
    #[multiversion::multiversion(targets(
        "x86_64+avx512vpopcntdq+avx512vl+popcnt",
        "x86_64+avx2+popcnt",
        "x86_64+sse4.2+popcnt",
    ))]
    pub fn slice_batch_calls_impl(source: &[u8], targets: &[&[u8]], out: &mut [u32]) {
        assert_eq!(targets.len(), out.len());
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            *dist = slice_impl(source, target);
        }
    }

    // MV batch with body manually inlined
    #[multiversion::multiversion(targets(
        "x86_64+avx512vpopcntdq+avx512vl+popcnt",
        "x86_64+avx2+popcnt",
        "x86_64+sse4.2+popcnt",
    ))]
    pub fn slice_batch_body_inlined(source: &[u8], targets: &[&[u8]], out: &mut [u32]) {
        assert_eq!(targets.len(), out.len());
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            assert_eq!(source.len(), target.len());
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

    // ========================================================================
    // ARRAY strategies
    // ========================================================================

    // Single array function with multiversion
    #[multiversion::multiversion(targets(
        "x86_64+avx512vpopcntdq+avx512vl+popcnt",
        "x86_64+avx2+popcnt",
        "x86_64+sse4.2+popcnt",
    ))]
    #[inline(always)]
    pub fn array_single_mv<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
        array_impl(a, b)
    }

    // MV batch calling #[inline(always)] impl (triggers gather - SLOW!)
    #[multiversion::multiversion(targets(
        "x86_64+avx512vpopcntdq+avx512vl+popcnt",
        "x86_64+avx2+popcnt",
        "x86_64+sse4.2+popcnt",
    ))]
    pub fn array_batch_calls_impl<const N: usize>(
        source: &[u8; N],
        targets: &[[u8; N]],
        out: &mut [u32],
    ) {
        assert_eq!(targets.len(), out.len());
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            *dist = array_impl(source, target);
        }
    }

    // MV batch with body inlined (triggers gather - SLOW!)
    #[multiversion::multiversion(targets(
        "x86_64+avx512vpopcntdq+avx512vl+popcnt",
        "x86_64+avx2+popcnt",
        "x86_64+sse4.2+popcnt",
    ))]
    pub fn array_batch_body_inlined<const N: usize>(
        source: &[u8; N],
        targets: &[[u8; N]],
        out: &mut [u32],
    ) {
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

    // MV batch calling MV single - prevents gather (FAST!)
    #[multiversion::multiversion(targets(
        "x86_64+avx512vpopcntdq+avx512vl+popcnt",
        "x86_64+avx2+popcnt",
        "x86_64+sse4.2+popcnt",
    ))]
    pub fn array_batch_calls_mv_single<const N: usize>(
        source: &[u8; N],
        targets: &[[u8; N]],
        out: &mut [u32],
    ) {
        assert_eq!(targets.len(), out.len());
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            *dist = array_single_mv(source, target);
        }
    }

    // No MV on batch, calls MV single (dispatch per iteration)
    pub fn array_batch_no_mv<const N: usize>(
        source: &[u8; N],
        targets: &[[u8; N]],
        out: &mut [u32],
    ) {
        assert_eq!(targets.len(), out.len());
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            *dist = array_single_mv(source, target);
        }
    }

    // MV batch with black_box to prevent gather
    #[multiversion::multiversion(targets(
        "x86_64+avx512vpopcntdq+avx512vl+popcnt",
        "x86_64+avx2+popcnt",
        "x86_64+sse4.2+popcnt",
    ))]
    pub fn array_batch_blackbox<const N: usize>(
        source: &[u8; N],
        targets: &[[u8; N]],
        out: &mut [u32],
    ) {
        assert_eq!(targets.len(), out.len());
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            let target: &[u8] = std::hint::black_box(target);

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
// Fallback for non-x86 or without multiversion feature
// ============================================================================

#[cfg(not(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
)))]
mod mv {
    use super::{array_impl, slice_impl};

    pub fn slice_batch_calls_impl(source: &[u8], targets: &[&[u8]], out: &mut [u32]) {
        assert_eq!(targets.len(), out.len());
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            *dist = slice_impl(source, target);
        }
    }

    pub fn slice_batch_body_inlined(source: &[u8], targets: &[&[u8]], out: &mut [u32]) {
        assert_eq!(targets.len(), out.len());
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            *dist = slice_impl(source, target);
        }
    }

    pub fn array_single_mv<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
        array_impl(a, b)
    }

    pub fn array_batch_calls_impl<const N: usize>(
        source: &[u8; N],
        targets: &[[u8; N]],
        out: &mut [u32],
    ) {
        assert_eq!(targets.len(), out.len());
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            *dist = array_impl(source, target);
        }
    }

    pub fn array_batch_body_inlined<const N: usize>(
        source: &[u8; N],
        targets: &[[u8; N]],
        out: &mut [u32],
    ) {
        assert_eq!(targets.len(), out.len());
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            *dist = array_impl(source, target);
        }
    }

    pub fn array_batch_calls_mv_single<const N: usize>(
        source: &[u8; N],
        targets: &[[u8; N]],
        out: &mut [u32],
    ) {
        assert_eq!(targets.len(), out.len());
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            *dist = array_single_mv(source, target);
        }
    }

    pub fn array_batch_no_mv<const N: usize>(
        source: &[u8; N],
        targets: &[[u8; N]],
        out: &mut [u32],
    ) {
        assert_eq!(targets.len(), out.len());
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            *dist = array_single_mv(source, target);
        }
    }

    pub fn array_batch_blackbox<const N: usize>(
        source: &[u8; N],
        targets: &[[u8; N]],
        out: &mut [u32],
    ) {
        assert_eq!(targets.len(), out.len());
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            *dist = array_impl(source, target);
        }
    }
}

// ============================================================================
// Benchmarks
// ============================================================================

fn benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiversion_placement");

    macro_rules! bench_size {
        ($size:expr) => {{
            let bits = format!("{}b", $size * 8);

            // Slice benchmarks
            {
                let source = random_bytes_vec($size);
                let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec($size)).collect();
                let targets_refs: Vec<&[u8]> = targets.iter().map(|v| v.as_slice()).collect();
                let mut out = vec![0u32; BATCH];

                group.bench_with_input(BenchmarkId::new("slice_calls_inline_fn", &bits), &$size, |bencher, _| {
                    bencher.iter(|| {
                        mv::slice_batch_calls_impl(black_box(&source), black_box(&targets_refs), &mut out);
                        black_box(out[0])
                    })
                });

                group.bench_with_input(BenchmarkId::new("slice_body_inlined", &bits), &$size, |bencher, _| {
                    bencher.iter(|| {
                        mv::slice_batch_body_inlined(black_box(&source), black_box(&targets_refs), &mut out);
                        black_box(out[0])
                    })
                });
            }

            // Array benchmarks
            {
                let source: [u8; $size] = random_bytes();
                let targets: Vec<[u8; $size]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];

                group.bench_with_input(BenchmarkId::new("array_calls_inline_fn_SLOW", &bits), &$size, |bencher, _| {
                    bencher.iter(|| {
                        mv::array_batch_calls_impl(black_box(&source), black_box(&targets), &mut out);
                        black_box(out[0])
                    })
                });

                group.bench_with_input(BenchmarkId::new("array_body_inlined_SLOW", &bits), &$size, |bencher, _| {
                    bencher.iter(|| {
                        mv::array_batch_body_inlined(black_box(&source), black_box(&targets), &mut out);
                        black_box(out[0])
                    })
                });

                group.bench_with_input(BenchmarkId::new("array_calls_mv_single", &bits), &$size, |bencher, _| {
                    bencher.iter(|| {
                        mv::array_batch_calls_mv_single(black_box(&source), black_box(&targets), &mut out);
                        black_box(out[0])
                    })
                });

                group.bench_with_input(BenchmarkId::new("array_plain_calls_mv_single", &bits), &$size, |bencher, _| {
                    bencher.iter(|| {
                        mv::array_batch_no_mv(black_box(&source), black_box(&targets), &mut out);
                        black_box(out[0])
                    })
                });

                group.bench_with_input(BenchmarkId::new("array_with_blackbox", &bits), &$size, |bencher, _| {
                    bencher.iter(|| {
                        mv::array_batch_blackbox(black_box(&source), black_box(&targets), &mut out);
                        black_box(out[0])
                    })
                });
            }
        }};
    }

    bench_size!(64);
    bench_size!(128);
    bench_size!(256);

    group.finish();
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);

//! Q8: Does inlining the comparison body into the batch loop help?
//!
//! Key questions:
//! - Is there any difference between calling a single-comparison function
//!   vs inlining the entire body into the batch loop?
//! - Does `#[inline(always)]` on the single function achieve the same effect?
//! - Are there cache/register benefits to having the body inlined?
//! - Does multiversion (runtime CPU dispatch) change the inlining calculus?
//!
//! Run with: cargo bench --bench q8_inlining
//! Run with multiversion: cargo bench --bench q8_inlining --features multiversion_x86

mod helpers;

use helpers::random_bytes_array;

fn main() {
    divan::main();
}

const BATCH: usize = 64;

// ============================================================================
// Approach 1: Batch calls out to single-comparison function
// ============================================================================

/// Single array comparison - marked inline(always) to encourage inlining.
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

/// Batch that calls the single function for each comparison.
#[inline]
fn batch_with_call<const N: usize>(source: &[u8; N], targets: &[[u8; N]], out: &mut [u32]) {
    assert_eq!(targets.len(), out.len());
    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        *dist = hamming_single(source, target);
    }
}

// ============================================================================
// Approach 2: Full body inlined into the batch loop
// ============================================================================

/// Batch with the comparison algorithm directly inlined in the loop body.
/// No separate function call - the compiler sees the full algorithm in context.
#[inline]
fn batch_body_inlined<const N: usize>(source: &[u8; N], targets: &[[u8; N]], out: &mut [u32]) {
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

// ============================================================================
// Approach 3: Single function without #[inline(always)] hint
// ============================================================================

/// Single comparison with only #[inline] (not always) - lets compiler decide.
#[inline]
fn hamming_single_hint_only<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
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

/// Batch that calls the #[inline] (not always) function.
#[inline]
fn batch_with_inline_hint<const N: usize>(source: &[u8; N], targets: &[[u8; N]], out: &mut [u32]) {
    assert_eq!(targets.len(), out.len());
    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        *dist = hamming_single_hint_only(source, target);
    }
}

// ============================================================================
// Benchmarks
// ============================================================================

mod array_batch {
    use super::*;
    use helpers::random_bytes;

    #[divan::bench(consts = [64, 96, 128, 256])]
    fn call_inline_always<const N: usize>(bencher: divan::Bencher) {
        let source: [u8; N] = random_bytes();
        let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            batch_with_call(&source, &targets, &mut out);
            out[0]
        });
    }

    #[divan::bench(consts = [64, 96, 128, 256])]
    fn body_inlined<const N: usize>(bencher: divan::Bencher) {
        let source: [u8; N] = random_bytes();
        let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            batch_body_inlined(&source, &targets, &mut out);
            out[0]
        });
    }

    #[divan::bench(consts = [64, 96, 128, 256])]
    fn call_inline_hint<const N: usize>(bencher: divan::Bencher) {
        let source: [u8; N] = random_bytes();
        let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            batch_with_inline_hint(&source, &targets, &mut out);
            out[0]
        });
    }
}

/// Test with larger batch sizes to see if inlining effects change.
mod large_batch {
    use super::*;
    use helpers::random_bytes;

    const LARGE_BATCH: usize = 256;

    #[divan::bench(consts = [128, 256])]
    fn call_inline_always<const N: usize>(bencher: divan::Bencher) {
        let source: [u8; N] = random_bytes();
        let targets: Vec<[u8; N]> = random_bytes_array(LARGE_BATCH);
        let mut out = vec![0u32; LARGE_BATCH];

        bencher.bench_local(|| {
            batch_with_call(&source, &targets, &mut out);
            out[0]
        });
    }

    #[divan::bench(consts = [128, 256])]
    fn body_inlined<const N: usize>(bencher: divan::Bencher) {
        let source: [u8; N] = random_bytes();
        let targets: Vec<[u8; N]> = random_bytes_array(LARGE_BATCH);
        let mut out = vec![0u32; LARGE_BATCH];

        bencher.bench_local(|| {
            batch_body_inlined(&source, &targets, &mut out);
            out[0]
        });
    }
}

// ============================================================================
// Multiversion implementations (x86 only with feature flag)
// ============================================================================

#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
mod multiversion_impl {
    /// Single comparison with multiversion - each call does CPU dispatch.
    #[multiversion::multiversion(targets(
        "x86_64+avx512vpopcntdq+avx512vl+popcnt",
        "x86_64+avx512bw+avx512vl+popcnt",
        "x86_64+avx2+popcnt",
        "x86_64+sse4.2+popcnt",
        "x86+avx2+popcnt",
        "x86+sse4.2+popcnt",
    ))]
    #[inline(always)]
    pub fn hamming_single_mv<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
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

    /// Batch that calls the multiversion single function for each comparison.
    /// Each iteration pays the CPU dispatch cost.
    #[inline]
    pub fn batch_with_mv_call<const N: usize>(
        source: &[u8; N],
        targets: &[[u8; N]],
        out: &mut [u32],
    ) {
        assert_eq!(targets.len(), out.len());
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            *dist = hamming_single_mv(source, target);
        }
    }

    /// Batch with multiversion on the whole batch function.
    /// CPU dispatch happens once for the entire batch.
    #[multiversion::multiversion(targets(
        "x86_64+avx512vpopcntdq+avx512vl+popcnt",
        "x86_64+avx512bw+avx512vl+popcnt",
        "x86_64+avx2+popcnt",
        "x86_64+sse4.2+popcnt",
        "x86+avx2+popcnt",
        "x86+sse4.2+popcnt",
    ))]
    #[inline]
    pub fn batch_mv_body_inlined<const N: usize>(
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
}

/// Multiversion benchmarks - only compiled on x86 with the feature.
#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
mod multiversion_bench {
    use super::helpers::{random_bytes, random_bytes_array};
    use super::multiversion_impl::*;
    use super::BATCH;

    #[divan::bench(consts = [64, 96, 128, 256])]
    fn mv_call_per_item<const N: usize>(bencher: divan::Bencher) {
        let source: [u8; N] = random_bytes();
        let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            batch_with_mv_call(&source, &targets, &mut out);
            out[0]
        });
    }

    #[divan::bench(consts = [64, 96, 128, 256])]
    fn mv_batch_inlined<const N: usize>(bencher: divan::Bencher) {
        let source: [u8; N] = random_bytes();
        let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            batch_mv_body_inlined(&source, &targets, &mut out);
            out[0]
        });
    }

    /// Large batch to amplify any per-call dispatch overhead.
    mod large_batch {
        use super::*;

        const LARGE_BATCH: usize = 256;

        #[divan::bench(consts = [128, 256])]
        fn mv_call_per_item<const N: usize>(bencher: divan::Bencher) {
            let source: [u8; N] = random_bytes();
            let targets: Vec<[u8; N]> = random_bytes_array(LARGE_BATCH);
            let mut out = vec![0u32; LARGE_BATCH];

            bencher.bench_local(|| {
                batch_with_mv_call(&source, &targets, &mut out);
                out[0]
            });
        }

        #[divan::bench(consts = [128, 256])]
        fn mv_batch_inlined<const N: usize>(bencher: divan::Bencher) {
            let source: [u8; N] = random_bytes();
            let targets: Vec<[u8; N]> = random_bytes_array(LARGE_BATCH);
            let mut out = vec![0u32; LARGE_BATCH];

            bencher.bench_local(|| {
                batch_mv_body_inlined(&source, &targets, &mut out);
                out[0]
            });
        }
    }
}

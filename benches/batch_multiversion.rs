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
//! Run with: cargo bench --bench batch_multiversion --features multiversion_x86
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
use helpers::{random_bytes, random_bytes_array};

#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
const BATCH: usize = 64;

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
        eprintln!("Run: cargo bench --bench batch_multiversion --features multiversion_x86");
    }
}

// ============================================================================
// Strategy 1: Dispatch per call
// ============================================================================

#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
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

#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
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

#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
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

#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[divan::bench(consts = [64, 96, 128, 256])]
fn dispatch_per_call<const N: usize>(bencher: divan::Bencher) {
    let source: [u8; N] = random_bytes();
    let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
    let mut out = vec![0u32; BATCH];

    bencher.bench_local(|| {
        dispatch_per_call::batch(&source, &targets, &mut out);
        out[0]
    });
}

#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[divan::bench(consts = [64, 96, 128, 256])]
fn dispatch_once_call<const N: usize>(bencher: divan::Bencher) {
    let source: [u8; N] = random_bytes();
    let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
    let mut out = vec![0u32; BATCH];

    bencher.bench_local(|| {
        dispatch_once_call::batch(&source, &targets, &mut out);
        out[0]
    });
}

#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[divan::bench(consts = [64, 96, 128, 256])]
fn dispatch_once_inlined<const N: usize>(bencher: divan::Bencher) {
    let source: [u8; N] = random_bytes();
    let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
    let mut out = vec![0u32; BATCH];

    bencher.bench_local(|| {
        dispatch_once_inlined::batch(&source, &targets, &mut out);
        out[0]
    });
}

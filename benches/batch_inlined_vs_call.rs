//! Compares inlining strategies in batch operations.
//!
//! Key insight: When processing batches, there's a tradeoff between:
//! - Calling a single-comparison function (cleaner, relies on compiler inlining)
//! - Inlining the entire comparison body (gives compiler full context)
//!
//! This benchmark measures whether `#[inline(always)]` achieves the same
//! optimization as manually inlining the loop body.
//!
//! Run with: cargo bench --bench batch_inlined_vs_call

mod helpers;

use helpers::{random_bytes, random_bytes_array};

fn main() {
    divan::main();
}

const BATCH: usize = 64;

// ============================================================================
// Approach 1: Single function with #[inline] hint - lets compiler decide
// ============================================================================

#[inline]
fn hamming_inline_hint<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
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

/// Batch that calls #[inline] function.
#[inline]
fn batch_with_inline_hint<const N: usize>(source: &[u8; N], targets: &[[u8; N]], out: &mut [u32]) {
    assert_eq!(targets.len(), out.len());
    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        *dist = hamming_inline_hint(source, target);
    }
}

// ============================================================================
// Approach 2: Single function with #[inline(always)] - forces inlining
// ============================================================================

#[inline(always)]
fn hamming_inline_always<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
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

/// Batch that calls #[inline(always)] function.
#[inline]
fn batch_with_inline_always<const N: usize>(source: &[u8; N], targets: &[[u8; N]], out: &mut [u32]) {
    assert_eq!(targets.len(), out.len());
    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        *dist = hamming_inline_always(source, target);
    }
}

// ============================================================================
// Approach 3: Comparison body fully inlined in batch function
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
// Benchmarks
// ============================================================================

#[divan::bench(consts = [64, 96, 128, 256])]
fn function_call<const N: usize>(bencher: divan::Bencher) {
    let source: [u8; N] = random_bytes();
    let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
    let mut out = vec![0u32; BATCH];

    bencher.bench_local(|| {
        batch_with_inline_hint(&source, &targets, &mut out);
        out[0]
    });
}

#[divan::bench(consts = [64, 96, 128, 256])]
fn inline_always<const N: usize>(bencher: divan::Bencher) {
    let source: [u8; N] = random_bytes();
    let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
    let mut out = vec![0u32; BATCH];

    bencher.bench_local(|| {
        batch_with_inline_always(&source, &targets, &mut out);
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

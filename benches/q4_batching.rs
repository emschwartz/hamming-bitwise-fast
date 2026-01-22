//! Q4: Does batching help? Fixed-size batch vs variable-size batch?
//!
//! Key questions:
//! - How much faster is batch processing vs looping single calls?
//! - Does the compiler optimize fixed-size batches better?
//! - What's the overhead of allocation patterns?
//!
//! Run with: cargo bench --bench q4_batching

mod helpers;

use helpers::{random_bytes, random_bytes_array};

fn main() {
    divan::main();
}

const BATCH: usize = 64;

// ============================================================================
// Core implementation (inlined from library)
// ============================================================================

/// Single array comparison using u64 chunks.
#[inline]
fn hamming_array<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
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

/// Batch with variable-size slice of targets.
#[inline]
fn hamming_batch<const N: usize>(source: &[u8; N], targets: &[[u8; N]], out: &mut [u32]) {
    assert_eq!(targets.len(), out.len());
    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        *dist = hamming_array(source, target);
    }
}

/// Batch with fixed-size array of targets.
#[inline]
fn hamming_batch_fixed<const N: usize, const B: usize>(
    source: &[u8; N],
    targets: &[[u8; N]; B],
    out: &mut [u32; B],
) {
    for i in 0..B {
        out[i] = hamming_array(source, &targets[i]);
    }
}

// ============================================================================
// Benchmarks
// ============================================================================

mod per_comparison {
    use super::*;

    #[divan::bench(consts = [64, 96, 128, 256])]
    fn single_pair<const N: usize>(bencher: divan::Bencher) {
        let a: [u8; N] = random_bytes();
        let b: [u8; N] = random_bytes();
        bencher.bench_local(|| hamming_array(&a, &b));
    }
}

mod batch_vs_loop {
    use super::*;

    #[divan::bench(consts = [64, 96, 128, 256])]
    fn batch_api<const N: usize>(bencher: divan::Bencher) {
        let source: [u8; N] = random_bytes();
        let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            hamming_batch(&source, &targets, &mut out);
            out[0]
        });
    }

    #[divan::bench(consts = [64, 96, 128, 256])]
    fn manual_loop<const N: usize>(bencher: divan::Bencher) {
        let source: [u8; N] = random_bytes();
        let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            for (target, dist) in targets.iter().zip(out.iter_mut()) {
                *dist = hamming_array(&source, target);
            }
            out[0]
        });
    }
}

mod fixed_vs_variable {
    use super::*;

    #[divan::bench(consts = [64, 96, 128, 256])]
    fn variable_size<const N: usize>(bencher: divan::Bencher) {
        let source: [u8; N] = random_bytes();
        let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            hamming_batch(&source, &targets, &mut out);
            out[0]
        });
    }

    #[divan::bench(consts = [64, 96, 128, 256])]
    fn fixed_size<const N: usize>(bencher: divan::Bencher) {
        let source: [u8; N] = random_bytes();
        let targets_vec: Vec<[u8; N]> = random_bytes_array(BATCH);
        let targets: [[u8; N]; BATCH] = targets_vec.try_into().unwrap();
        let mut out = [0u32; BATCH];

        bencher.bench_local(|| {
            hamming_batch_fixed(&source, &targets, &mut out);
            out[0]
        });
    }
}

mod allocation {
    use super::*;

    #[divan::bench(consts = [64, 96, 128, 256])]
    fn alloc_per_batch<const N: usize>(bencher: divan::Bencher) {
        let source: [u8; N] = random_bytes();
        let targets: Vec<[u8; N]> = random_bytes_array(BATCH);

        bencher.bench_local(|| {
            let mut out = vec![0u32; BATCH];
            hamming_batch(&source, &targets, &mut out);
            out[0]
        });
    }

    #[divan::bench(consts = [64, 96, 128, 256])]
    fn preallocated<const N: usize>(bencher: divan::Bencher) {
        let source: [u8; N] = random_bytes();
        let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            hamming_batch(&source, &targets, &mut out);
            out[0]
        });
    }
}

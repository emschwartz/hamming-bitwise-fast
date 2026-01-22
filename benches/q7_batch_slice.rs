//! Q7: How does batch slice compare to individual slice calls and competitors?
//!
//! Key questions:
//! - Does a batch API provide speedup over a loop?
//! - How does our slice implementation compare to competitor crates?
//!
//! Run with: cargo bench --bench q7_batch_slice

mod helpers;

use helpers::random_bytes_vec;

fn main() {
    divan::main();
}

const BATCH: usize = 64;

// ============================================================================
// Implementations
// ============================================================================

/// Single slice comparison using u64 chunks.
#[inline]
fn hamming_slice(a: &[u8], b: &[u8]) -> u32 {
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

/// Batch slice comparison.
#[inline]
fn hamming_slice_batch(source: &[u8], targets: &[&[u8]], out: &mut [u32]) {
    assert_eq!(targets.len(), out.len());
    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        *dist = hamming_slice(source, target);
    }
}

// ============================================================================
// Benchmarks: Batch vs loop
// ============================================================================

mod batch_vs_loop {
    use super::*;

    #[divan::bench(args = [64, 96, 128, 256])]
    fn batch_api(bencher: divan::Bencher, bytes: usize) {
        let source = random_bytes_vec(bytes);
        let targets_owned: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(bytes)).collect();
        let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            hamming_slice_batch(&source, &targets, &mut out);
            out[0]
        });
    }

    #[divan::bench(args = [64, 96, 128, 256])]
    fn manual_loop(bencher: divan::Bencher, bytes: usize) {
        let source = random_bytes_vec(bytes);
        let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(bytes)).collect();
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            for (i, target) in targets.iter().enumerate() {
                out[i] = hamming_slice(&source, target);
            }
            out[0]
        });
    }
}

// ============================================================================
// Benchmarks: Comparison with competitors
// ============================================================================

mod vs_competitors {
    use super::*;

    #[divan::bench(args = [64, 96, 128, 256])]
    fn ours(bencher: divan::Bencher, bytes: usize) {
        let source = random_bytes_vec(bytes);
        let targets_owned: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(bytes)).collect();
        let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            hamming_slice_batch(&source, &targets, &mut out);
            out[0]
        });
    }

    #[divan::bench(args = [64, 96, 128, 256])]
    fn simsimd(bencher: divan::Bencher, bytes: usize) {
        let source = random_bytes_vec(bytes);
        let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(bytes)).collect();
        let mut out = vec![0f64; BATCH];

        bencher.bench_local(|| {
            for (i, target) in targets.iter().enumerate() {
                out[i] = simsimd::BinarySimilarity::hamming(&source, target).unwrap_or(0.0);
            }
            out[0] as u64
        });
    }

    #[divan::bench(args = [64, 96, 128, 256])]
    fn hamming_crate(bencher: divan::Bencher, bytes: usize) {
        let source = random_bytes_vec(bytes);
        let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(bytes)).collect();
        let mut out = vec![0u64; BATCH];

        bencher.bench_local(|| {
            for (i, target) in targets.iter().enumerate() {
                out[i] = hamming::distance_fast(&source, target).unwrap_or(0);
            }
            out[0]
        });
    }

    #[divan::bench(args = [64, 96, 128, 256])]
    fn triple_accel(bencher: divan::Bencher, bytes: usize) {
        let source = random_bytes_vec(bytes);
        let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(bytes)).collect();
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            for (i, target) in targets.iter().enumerate() {
                out[i] = triple_accel::hamming(&source, target);
            }
            out[0]
        });
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[divan::bench(args = [64, 96, 128, 256])]
    fn hamming_rs(bencher: divan::Bencher, bytes: usize) {
        let source = random_bytes_vec(bytes);
        let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(bytes)).collect();
        let mut out = vec![0u64; BATCH];

        bencher.bench_local(|| {
            for (i, target) in targets.iter().enumerate() {
                out[i] = hamming_rs::distance_faster(&source, target);
            }
            out[0]
        });
    }
}

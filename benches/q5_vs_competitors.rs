//! Q5: How does our library compare to other Hamming distance crates?
//!
//! Key questions:
//! - How do we compare to simsimd, hamming, triple_accel, hamming_rs?
//! - What's the speedup from batch operations?
//!
//! Run with: cargo bench --bench q5_vs_competitors

mod helpers;

use hamming_bitwise_fast::{
    hamming_bitwise_array, hamming_bitwise_array_batch, hamming_bitwise_slice,
    hamming_bitwise_slice_batch,
};
use helpers::{random_bytes, random_bytes_array, random_bytes_vec};

fn main() {
    divan::main();
}

const BATCH: usize = 1000;

// ============================================================================
// Alternative implementation to benchmark
// ============================================================================

/// Original v1 implementation: u64 chunked processing without multiversion.
#[inline]
fn hamming_bitwise_slice_v1(x: &[u8], y: &[u8]) -> u32 {
    assert_eq!(x.len(), y.len());

    let mut distance = x
        .chunks_exact(8)
        .zip(y.chunks_exact(8))
        .map(|(x_chunk, y_chunk)| {
            let x_val = u64::from_ne_bytes(x_chunk.try_into().unwrap());
            let y_val = u64::from_ne_bytes(y_chunk.try_into().unwrap());
            (x_val ^ y_val).count_ones()
        })
        .sum::<u32>();

    for (x_byte, y_byte) in x
        .chunks_exact(8)
        .remainder()
        .iter()
        .zip(y.chunks_exact(8).remainder())
    {
        distance += (x_byte ^ y_byte).count_ones();
    }

    distance
}

// ============================================================================
// Benchmarks: Single comparison
// ============================================================================

mod single {
    use super::*;

    #[divan::bench(consts = [64, 96, 128, 256])]
    fn ours_array<const N: usize>(bencher: divan::Bencher) {
        let a: [u8; N] = random_bytes();
        let b: [u8; N] = random_bytes();
        bencher.bench_local(|| hamming_bitwise_array(&a, &b));
    }

    #[divan::bench(args = [64, 96, 128, 256])]
    fn ours_slice(bencher: divan::Bencher, bytes: usize) {
        let a = random_bytes_vec(bytes);
        let b = random_bytes_vec(bytes);
        bencher.bench_local(|| hamming_bitwise_slice(&a, &b));
    }

    #[divan::bench(args = [64, 96, 128, 256])]
    fn ours_slice_v1(bencher: divan::Bencher, bytes: usize) {
        let a = random_bytes_vec(bytes);
        let b = random_bytes_vec(bytes);
        bencher.bench_local(|| hamming_bitwise_slice_v1(&a, &b));
    }

    #[divan::bench(args = [64, 96, 128, 256])]
    fn simsimd(bencher: divan::Bencher, bytes: usize) {
        let a = random_bytes_vec(bytes);
        let b = random_bytes_vec(bytes);
        bencher.bench_local(|| simsimd::BinarySimilarity::hamming(&a, &b));
    }

    #[divan::bench(args = [64, 96, 128, 256])]
    fn hamming_crate(bencher: divan::Bencher, bytes: usize) {
        let a = random_bytes_vec(bytes);
        let b = random_bytes_vec(bytes);
        bencher.bench_local(|| hamming::distance_fast(&a, &b));
    }

    #[divan::bench(args = [64, 96, 128, 256])]
    fn triple_accel(bencher: divan::Bencher, bytes: usize) {
        let a = random_bytes_vec(bytes);
        let b = random_bytes_vec(bytes);
        bencher.bench_local(|| triple_accel::hamming(&a, &b));
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[divan::bench(args = [64, 96, 128, 256])]
    fn hamming_rs(bencher: divan::Bencher, bytes: usize) {
        let a = random_bytes_vec(bytes);
        let b = random_bytes_vec(bytes);
        bencher.bench_local(|| hamming_rs::distance_faster(&a, &b));
    }
}

// ============================================================================
// Benchmarks: Batch (1000 comparisons)
// ============================================================================

mod batch_1000 {
    use super::*;

    #[divan::bench(consts = [64, 96, 128, 256])]
    fn ours_array_batch<const N: usize>(bencher: divan::Bencher) {
        let source: [u8; N] = random_bytes();
        let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            hamming_bitwise_array_batch(&source, &targets, &mut out);
            out[0]
        });
    }

    #[divan::bench(consts = [64, 96, 128, 256])]
    fn ours_array_loop<const N: usize>(bencher: divan::Bencher) {
        let source: [u8; N] = random_bytes();
        let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            for (i, target) in targets.iter().enumerate() {
                out[i] = hamming_bitwise_array(&source, target);
            }
            out[0]
        });
    }

    #[divan::bench(args = [64, 96, 128, 256])]
    fn ours_slice_batch(bencher: divan::Bencher, bytes: usize) {
        let source = random_bytes_vec(bytes);
        let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(bytes)).collect();
        let targets_refs: Vec<&[u8]> = targets.iter().map(|v| v.as_slice()).collect();
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            hamming_bitwise_slice_batch(&source, &targets_refs, &mut out);
            out[0]
        });
    }

    #[divan::bench(args = [64, 96, 128, 256])]
    fn ours_slice_loop(bencher: divan::Bencher, bytes: usize) {
        let source = random_bytes_vec(bytes);
        let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(bytes)).collect();
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            for (i, target) in targets.iter().enumerate() {
                out[i] = hamming_bitwise_slice(&source, target);
            }
            out[0]
        });
    }

    #[divan::bench(args = [64, 96, 128, 256])]
    fn ours_slice_v1_loop(bencher: divan::Bencher, bytes: usize) {
        let source = random_bytes_vec(bytes);
        let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(bytes)).collect();
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            for (i, target) in targets.iter().enumerate() {
                out[i] = hamming_bitwise_slice_v1(&source, target);
            }
            out[0]
        });
    }

    #[divan::bench(args = [64, 96, 128, 256])]
    fn simsimd_loop(bencher: divan::Bencher, bytes: usize) {
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
    fn hamming_crate_loop(bencher: divan::Bencher, bytes: usize) {
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
    fn triple_accel_loop(bencher: divan::Bencher, bytes: usize) {
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
    fn hamming_rs_loop(bencher: divan::Bencher, bytes: usize) {
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

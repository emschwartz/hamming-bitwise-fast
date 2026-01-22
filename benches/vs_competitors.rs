//! Compares this crate vs other Hamming distance implementations.
//!
//! Competitors:
//! - simsimd: SIMD-optimized similarity functions
//! - hamming: Pure Rust implementation
//! - triple_accel: SIMD-accelerated string metrics
//! - hamming_rs: x86-only AVX2/SSE implementation
//!
//! Run with: cargo bench --bench vs_competitors

mod helpers;

use hamming_bitwise_fast;
use helpers::{random_bytes, random_bytes_array, random_bytes_vec};

fn main() {
    divan::main();
}

const BATCH: usize = 1000;

// ============================================================================
// Single comparison benchmarks
// ============================================================================

mod single {
    use super::*;

    #[divan::bench(consts = [64, 96, 128, 256])]
    fn hamming_bitwise_array<const N: usize>(bencher: divan::Bencher) {
        let a: [u8; N] = random_bytes();
        let b: [u8; N] = random_bytes();
        bencher.bench_local(|| hamming_bitwise_fast::hamming_bitwise_array(&a, &b));
    }

    #[divan::bench(args = [64, 96, 128, 256])]
    fn hamming_bitwise_slice(bencher: divan::Bencher, bytes: usize) {
        let a = random_bytes_vec(bytes);
        let b = random_bytes_vec(bytes);
        bencher.bench_local(|| hamming_bitwise_fast::hamming_bitwise_slice(&a, &b));
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
// Batch comparison benchmarks (1000 comparisons)
// ============================================================================

mod batch {
    use super::*;

    #[divan::bench(consts = [64, 96, 128, 256])]
    fn hamming_bitwise_array_batch<const N: usize>(bencher: divan::Bencher) {
        let source: [u8; N] = random_bytes();
        let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            hamming_bitwise_fast::hamming_bitwise_array_batch(&source, &targets, &mut out);
            out[0]
        });
    }

    #[divan::bench(consts = [64, 96, 128, 256])]
    fn hamming_bitwise_array_loop<const N: usize>(bencher: divan::Bencher) {
        let source: [u8; N] = random_bytes();
        let targets: Vec<[u8; N]> = random_bytes_array(BATCH);
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            for (i, target) in targets.iter().enumerate() {
                out[i] = hamming_bitwise_fast::hamming_bitwise_array(&source, target);
            }
            out[0]
        });
    }

    #[divan::bench(args = [64, 96, 128, 256])]
    fn hamming_bitwise_slice_batch(bencher: divan::Bencher, bytes: usize) {
        let source = random_bytes_vec(bytes);
        let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(bytes)).collect();
        let targets_refs: Vec<&[u8]> = targets.iter().map(|v| v.as_slice()).collect();
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            hamming_bitwise_fast::hamming_bitwise_slice_batch(&source, &targets_refs, &mut out);
            out[0]
        });
    }

    #[divan::bench(args = [64, 96, 128, 256])]
    fn hamming_bitwise_slice_loop(bencher: divan::Bencher, bytes: usize) {
        let source = random_bytes_vec(bytes);
        let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(bytes)).collect();
        let mut out = vec![0u32; BATCH];

        bencher.bench_local(|| {
            for (i, target) in targets.iter().enumerate() {
                out[i] = hamming_bitwise_fast::hamming_bitwise_slice(&source, target);
            }
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

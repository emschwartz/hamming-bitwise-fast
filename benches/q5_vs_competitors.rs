//! Q5: How much better are our optimizations than the original and other crates?
//!
//! Key questions:
//! - How does our new hamming<N> compare to the original hamming_bitwise_fast?
//! - How do we compare to other Hamming distance crates (simsimd, hamming, triple_accel)?
//! - What's the speedup from all optimizations combined (batch + arrays)?
//!
//! Run with: cargo bench --bench q5_vs_competitors
//! Filter by size: cargo bench --bench q5_vs_competitors -- 128

mod helpers;

use hamming_bitwise_fast::{hamming, hamming_batch, hamming_bitwise_fast};
use helpers::*;

fn main() {
    divan::main();
}

// ============================================================================
// Single comparison: Our APIs vs external crates
// Sizes in bytes: 64=512bit, 96=768bit, 128=1024bit, 256=2048bit
// ============================================================================

#[divan::bench_group(name = "single")]
mod single_comparison {
    use super::*;

    // --- Our implementations ---

    /// Original slice-based API
    #[divan::bench(args = [64, 96, 128, 256], name = "ours_slice")]
    fn ours_slice_api(bencher: divan::Bencher, size: usize) {
        let a = random_bytes_vec(size);
        let b = random_bytes_vec(size);

        bencher.bench_local(|| {
            hamming_bitwise_fast(divan::black_box(&a), divan::black_box(&b))
        });
    }

    /// New const-generic array API
    #[divan::bench(consts = [8, 12, 16, 32], name = "ours_array")]
    fn ours_array_api<const N: usize>(bencher: divan::Bencher) {
        let a: Embedding<N> = random_embedding();
        let b: Embedding<N> = random_embedding();

        bencher.bench_local(|| {
            hamming(divan::black_box(&a), divan::black_box(&b))
        });
    }

    // --- External crates ---

    /// simsimd: SIMD intrinsics
    #[divan::bench(args = [64, 96, 128, 256], name = "simsimd")]
    fn simsimd(bencher: divan::Bencher, size: usize) {
        let a = random_bytes_vec(size);
        let b = random_bytes_vec(size);

        bencher.bench_local(|| {
            simsimd::BinarySimilarity::hamming(divan::black_box(&a), divan::black_box(&b))
        });
    }

    /// hamming crate: Pure Rust
    #[divan::bench(args = [64, 96, 128, 256], name = "hamming_crate")]
    fn hamming_crate(bencher: divan::Bencher, size: usize) {
        let a = random_bytes_vec(size);
        let b = random_bytes_vec(size);

        bencher.bench_local(|| {
            hamming::distance_fast(divan::black_box(&a), divan::black_box(&b))
        });
    }

    /// triple_accel: SIMD-accelerated
    #[divan::bench(args = [64, 96, 128, 256], name = "triple_accel")]
    fn triple_accel(bencher: divan::Bencher, size: usize) {
        let a = random_bytes_vec(size);
        let b = random_bytes_vec(size);

        bencher.bench_local(|| {
            triple_accel::hamming(divan::black_box(&a), divan::black_box(&b))
        });
    }

    /// hamming_rs: x86-specific AVX2/AVX512
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[divan::bench(args = [64, 96, 128, 256], name = "hamming_rs")]
    fn hamming_rs(bencher: divan::Bencher, size: usize) {
        let a = random_bytes_vec(size);
        let b = random_bytes_vec(size);

        bencher.bench_local(|| {
            hamming_rs::distance_faster(divan::black_box(&a), divan::black_box(&b))
        });
    }
}

// ============================================================================
// Batch comparison: Our batch API vs competitors looping
// All preallocated for fair comparison. 64 element batches.
// ============================================================================

#[divan::bench_group(name = "batch_64")]
mod batch_comparison {
    use super::*;

    const BATCH: usize = 64;

    /// Our batch API
    #[divan::bench(consts = [8, 12, 16, 32], name = "ours_batch")]
    fn ours_batch<const N: usize>(bencher: divan::Bencher) {
        let source: Embedding<N> = random_embedding();
        let targets = random_embeddings::<N>(BATCH);
        let mut out = vec![0u32; BATCH];

        bencher
            .counter(divan::counter::ItemsCount::new(BATCH))
            .bench_local(|| {
                hamming_batch(
                    divan::black_box(&source),
                    divan::black_box(&targets),
                    &mut out,
                );
                divan::black_box(out[0])
            });
    }

    /// Our loop with array API
    #[divan::bench(consts = [8, 12, 16, 32], name = "ours_loop")]
    fn ours_loop<const N: usize>(bencher: divan::Bencher) {
        let source: Embedding<N> = random_embedding();
        let targets = random_embeddings::<N>(BATCH);
        let mut out = vec![0u32; BATCH];

        bencher
            .counter(divan::counter::ItemsCount::new(BATCH))
            .bench_local(|| {
                for (i, target) in targets.iter().enumerate() {
                    out[i] = hamming(divan::black_box(&source), divan::black_box(target));
                }
                divan::black_box(out[0])
            });
    }

    /// simsimd loop
    #[divan::bench(args = [64, 96, 128, 256], name = "simsimd_loop")]
    fn simsimd_loop(bencher: divan::Bencher, size: usize) {
        let source = random_bytes_vec(size);
        let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(size)).collect();
        let mut out = vec![0f64; BATCH];

        bencher
            .counter(divan::counter::ItemsCount::new(BATCH))
            .bench_local(|| {
                for (i, target) in targets.iter().enumerate() {
                    out[i] = simsimd::BinarySimilarity::hamming(
                        divan::black_box(&source),
                        divan::black_box(target),
                    )
                    .unwrap_or(0.0);
                }
                divan::black_box(out[0] as u64)
            });
    }

    /// triple_accel loop
    #[divan::bench(args = [64, 96, 128, 256], name = "triple_accel_loop")]
    fn triple_accel_loop(bencher: divan::Bencher, size: usize) {
        let source = random_bytes_vec(size);
        let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(size)).collect();
        let mut out = vec![0u32; BATCH];

        bencher
            .counter(divan::counter::ItemsCount::new(BATCH))
            .bench_local(|| {
                for (i, target) in targets.iter().enumerate() {
                    out[i] = triple_accel::hamming(
                        divan::black_box(&source),
                        divan::black_box(target),
                    );
                }
                divan::black_box(out[0])
            });
    }

    /// hamming crate loop
    #[divan::bench(args = [64, 96, 128, 256], name = "hamming_crate_loop")]
    fn hamming_crate_loop(bencher: divan::Bencher, size: usize) {
        let source = random_bytes_vec(size);
        let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(size)).collect();
        let mut out = vec![0u64; BATCH];

        bencher
            .counter(divan::counter::ItemsCount::new(BATCH))
            .bench_local(|| {
                for (i, target) in targets.iter().enumerate() {
                    out[i] = hamming::distance_fast(
                        divan::black_box(&source),
                        divan::black_box(target),
                    )
                    .unwrap_or(0);
                }
                divan::black_box(out[0])
            });
    }

    /// hamming_rs loop (x86 only)
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[divan::bench(args = [64, 96, 128, 256], name = "hamming_rs_loop")]
    fn hamming_rs_loop(bencher: divan::Bencher, size: usize) {
        let source = random_bytes_vec(size);
        let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(size)).collect();
        let mut out = vec![0usize; BATCH];

        bencher
            .counter(divan::counter::ItemsCount::new(BATCH))
            .bench_local(|| {
                for (i, target) in targets.iter().enumerate() {
                    out[i] = hamming_rs::distance_faster(
                        divan::black_box(&source),
                        divan::black_box(target),
                    );
                }
                divan::black_box(out[0])
            });
    }
}

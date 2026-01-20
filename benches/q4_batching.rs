//! Q4: Does batching help? Fixed-size batch vs variable-size batch?
//!
//! Key questions:
//! - How much faster is batch processing vs looping single calls?
//! - Does the compiler optimize fixed-size batches better?
//! - What's the overhead of allocation patterns?
//!
//! Times shown are PER COMPARISON for easy comparison with Q1.
//!
//! Run with: cargo bench --bench q4_batching
//! Filter by size: cargo bench --bench q4_batching -- 1024
//! Filter by batch: cargo bench --bench q4_batching -- batch_64

mod helpers;

use hamming_bitwise_fast::hamming_batch;
use helpers::*;
use std::cell::Cell;

fn main() {
    divan::main();
}

// ============================================================================
// Per-comparison time: cycling through targets of different batch sizes
// Embedding sizes: 8=512bit, 12=768bit, 16=1024bit, 32=2048bit (in u64s)
// ============================================================================

#[divan::bench_group]
mod per_comparison {
    use super::*;

    /// Single pair (no batch context) - baseline
    #[divan::bench(consts = [8, 12, 16, 32], name = "single_pair")]
    fn single_pair<const N: usize>(bencher: divan::Bencher) {
        let a: Embedding<N> = random_embedding();
        let b: Embedding<N> = random_embedding();

        bencher.bench_local(|| {
            hamming_bitwise_fast::hamming(divan::black_box(&a), divan::black_box(&b))
        });
    }

    /// One comparison from batch of 64 (fits in L1)
    #[divan::bench(consts = [8, 12, 16, 32], name = "batch_64")]
    fn from_batch_64<const N: usize>(bencher: divan::Bencher) {
        let source: Embedding<N> = random_embedding();
        let targets = random_embeddings::<N>(64);
        let idx = Cell::new(0usize);

        bencher.bench_local(|| {
            let i = idx.get();
            let result = hamming_bitwise_fast::hamming(
                divan::black_box(&source),
                divan::black_box(&targets[i]),
            );
            idx.set((i + 1) % 64);
            result
        });
    }

    /// One comparison from batch of 256 (fits in L1)
    #[divan::bench(consts = [8, 12, 16, 32], name = "batch_256")]
    fn from_batch_256<const N: usize>(bencher: divan::Bencher) {
        let source: Embedding<N> = random_embedding();
        let targets = random_embeddings::<N>(256);
        let idx = Cell::new(0usize);

        bencher.bench_local(|| {
            let i = idx.get();
            let result = hamming_bitwise_fast::hamming(
                divan::black_box(&source),
                divan::black_box(&targets[i]),
            );
            idx.set((i + 1) % 256);
            result
        });
    }

    /// One comparison from batch of 1024 (may spill to L2)
    #[divan::bench(consts = [8, 12, 16, 32], name = "batch_1024")]
    fn from_batch_1024<const N: usize>(bencher: divan::Bencher) {
        let source: Embedding<N> = random_embedding();
        let targets = random_embeddings::<N>(1024);
        let idx = Cell::new(0usize);

        bencher.bench_local(|| {
            let i = idx.get();
            let result = hamming_bitwise_fast::hamming(
                divan::black_box(&source),
                divan::black_box(&targets[i]),
            );
            idx.set((i + 1) % 1024);
            result
        });
    }
}

// ============================================================================
// Batch API throughput (items/sec shown, divide time by batch_size for per-item)
// ============================================================================

#[divan::bench_group]
mod batch_api {
    use super::*;

    #[divan::bench(consts = [8, 12, 16, 32], name = "batch_64")]
    fn batch_64<const N: usize>(bencher: divan::Bencher) {
        let source: Embedding<N> = random_embedding();
        let targets = random_embeddings::<N>(64);
        let mut out = vec![0u32; 64];

        bencher
            .counter(divan::counter::ItemsCount::new(64usize))
            .bench_local(|| {
                hamming_batch(
                    divan::black_box(&source),
                    divan::black_box(&targets),
                    &mut out,
                );
                divan::black_box(out[0])
            });
    }

    #[divan::bench(consts = [8, 12, 16, 32], name = "batch_256")]
    fn batch_256<const N: usize>(bencher: divan::Bencher) {
        let source: Embedding<N> = random_embedding();
        let targets = random_embeddings::<N>(256);
        let mut out = vec![0u32; 256];

        bencher
            .counter(divan::counter::ItemsCount::new(256usize))
            .bench_local(|| {
                hamming_batch(
                    divan::black_box(&source),
                    divan::black_box(&targets),
                    &mut out,
                );
                divan::black_box(out[0])
            });
    }
}

// ============================================================================
// Fixed vs variable batch size (64 elements)
// ============================================================================

#[divan::bench_group]
mod fixed_vs_variable {
    use super::*;

    const BATCH: usize = 64;

    #[divan::bench(consts = [8, 12, 16, 32], name = "variable")]
    fn variable<const N: usize>(bencher: divan::Bencher) {
        let source: Embedding<N> = random_embedding();
        let targets = random_embeddings::<N>(BATCH);
        let mut out = vec![0u32; BATCH];

        bencher
            .counter(divan::counter::ItemsCount::new(BATCH))
            .bench_local(|| {
                hamming_batch_variable(
                    divan::black_box(&source),
                    divan::black_box(&targets),
                    &mut out,
                );
                divan::black_box(out[0])
            });
    }

    #[divan::bench(consts = [8, 12, 16, 32], name = "fixed")]
    fn fixed<const N: usize>(bencher: divan::Bencher) {
        let source: Embedding<N> = random_embedding();
        let targets_vec = random_embeddings::<N>(BATCH);
        let targets: [Embedding<N>; BATCH] = targets_vec.try_into().unwrap();
        let mut out = [0u32; BATCH];

        bencher
            .counter(divan::counter::ItemsCount::new(BATCH))
            .bench_local(|| {
                hamming_batch_fixed(
                    divan::black_box(&source),
                    divan::black_box(&targets),
                    &mut out,
                );
                divan::black_box(out[0])
            });
    }
}

// ============================================================================
// Allocation patterns (64 element batches)
// ============================================================================

#[divan::bench_group]
mod allocation {
    use super::*;

    const BATCH: usize = 64;

    #[divan::bench(consts = [8, 12, 16, 32], name = "alloc_per_batch")]
    fn alloc_per_batch<const N: usize>(bencher: divan::Bencher) {
        let source: Embedding<N> = random_embedding();
        let targets = random_embeddings::<N>(BATCH);

        bencher
            .counter(divan::counter::ItemsCount::new(BATCH))
            .bench_local(|| {
                let mut out = vec![0u32; BATCH];
                hamming_batch(
                    divan::black_box(&source),
                    divan::black_box(&targets),
                    &mut out,
                );
                divan::black_box(out[0])
            });
    }

    #[divan::bench(consts = [8, 12, 16, 32], name = "preallocated")]
    fn preallocated<const N: usize>(bencher: divan::Bencher) {
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
}

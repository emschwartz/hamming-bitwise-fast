//! Shared test data generation for benchmarks.

#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, Ordering};

use criterion::BatchSize;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const BASE_SEED: u64 = 42;
static CALL_COUNTER: AtomicU64 = AtomicU64::new(0);

fn seeded_rng() -> StdRng {
    let offset = CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
    StdRng::seed_from_u64(BASE_SEED.wrapping_add(offset))
}

pub fn random_bytes<const N: usize>() -> [u8; N] {
    let mut rng = seeded_rng();
    std::array::from_fn(|_| rng.random())
}

pub fn random_bytes_array<const N: usize>(count: usize) -> Vec<[u8; N]> {
    let mut rng = seeded_rng();
    (0..count)
        .map(|_| std::array::from_fn(|_| rng.random()))
        .collect()
}

pub fn random_bytes_vec(size: usize) -> Vec<u8> {
    let mut rng = seeded_rng();
    (0..size).map(|_| rng.random()).collect()
}

/// Choose a batch size that keeps all input data within L1 data cache.
///
/// `iter_batched_ref` pre-allocates a Vec of all inputs for the batch.
/// With `SmallInput` (~1000 elements), large inputs (e.g., 128-byte arrays)
/// create batches of 256KB+ that overflow L1 cache, adding ~3-4ns of L2
/// latency per iteration and inflating benchmark results.
///
/// This function computes the number of iterations that fit the total input
/// data within a conservative 32KB L1 estimate (Intel CPUs; AMD Zen 4+ has
/// 64KB but we use the smaller value for portability).
pub fn l1_batch_size(input_bytes: usize) -> BatchSize {
    const L1_BYTES: usize = 32 * 1024;
    let iters = (L1_BYTES / input_bytes).max(1);
    BatchSize::NumIterations(iters as u64)
}

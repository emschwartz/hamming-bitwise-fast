//! Shared test data generation for benchmarks.

#![allow(dead_code)]

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const SEED: u64 = 42;

fn seeded_rng() -> StdRng {
    StdRng::seed_from_u64(SEED)
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

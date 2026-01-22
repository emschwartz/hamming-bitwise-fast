//! Shared test data generation for benchmarks.

#![allow(dead_code)]

use rand::Rng;

pub fn random_bytes<const N: usize>() -> [u8; N] {
    let mut rng = rand::rng();
    std::array::from_fn(|_| rng.random())
}

pub fn random_bytes_array<const N: usize>(count: usize) -> Vec<[u8; N]> {
    (0..count).map(|_| random_bytes()).collect()
}

pub fn random_bytes_vec(size: usize) -> Vec<u8> {
    let mut rng = rand::rng();
    (0..size).map(|_| rng.random()).collect()
}

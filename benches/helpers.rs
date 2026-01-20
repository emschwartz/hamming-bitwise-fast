//! Shared helpers for criterion benchmarks.
//!
//! This module provides test data generation and implementation variants
//! used across all benchmark files.

#![allow(dead_code)]

use rand::Rng;

// ============================================================================
// Wrapper for bit sizes that displays nicely in benchmark output
// ============================================================================

#[derive(Clone, Copy)]
pub struct BitSize {
    pub bits: usize,
}

impl BitSize {
    pub const fn new(bits: usize) -> Self {
        Self { bits }
    }

    pub const fn bytes(&self) -> usize {
        self.bits / 8
    }
}

impl std::fmt::Display for BitSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}b", self.bits)
    }
}

/// Standard embedding bit sizes for benchmarks.
pub const BIT_SIZES: [BitSize; 4] = [
    BitSize::new(512),
    BitSize::new(768),
    BitSize::new(1024),
    BitSize::new(2048),
];

// ============================================================================
// Test data generation
// ============================================================================

pub fn random_bytes<const N: usize>() -> [u8; N] {
    let mut rng = rand::thread_rng();
    std::array::from_fn(|_| rng.gen())
}

pub fn random_bytes_array<const N: usize>(count: usize) -> Vec<[u8; N]> {
    (0..count).map(|_| random_bytes()).collect()
}

pub fn random_bytes_vec(size: usize) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen()).collect()
}

// ============================================================================
// u8 array implementations (for Q1: data type comparison)
// ============================================================================

/// Hamming distance on u8 arrays using iterator (byte-by-byte).
#[inline]
pub fn hamming_u8_iter<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Hamming distance on u8 arrays using for loop.
#[inline]
pub fn hamming_u8_for<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    let mut dist = 0u32;
    for i in 0..N {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

/// Hamming distance on u8 arrays using chunks_exact(8) to process as u64.
/// This is the strategy that enables AVX-512 VPOPCNTDQ on x86.
#[inline]
pub fn hamming_u8_chunks<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    a.chunks_exact(8)
        .zip(b.chunks_exact(8))
        .map(|(a_chunk, b_chunk)| {
            let a_val = u64::from_ne_bytes(a_chunk.try_into().unwrap());
            let b_val = u64::from_ne_bytes(b_chunk.try_into().unwrap());
            (a_val ^ b_val).count_ones()
        })
        .sum()
}

/// Hamming distance on u8 arrays using chunks_exact(8) with remainder handling.
#[inline]
pub fn hamming_u8_chunks_with_remainder<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    let mut dist: u32 = a
        .chunks_exact(8)
        .zip(b.chunks_exact(8))
        .map(|(a_chunk, b_chunk)| {
            let a_val = u64::from_ne_bytes(a_chunk.try_into().unwrap());
            let b_val = u64::from_ne_bytes(b_chunk.try_into().unwrap());
            (a_val ^ b_val).count_ones()
        })
        .sum();

    // Handle remainder bytes
    for (x, y) in a
        .chunks_exact(8)
        .remainder()
        .iter()
        .zip(b.chunks_exact(8).remainder())
    {
        dist += (x ^ y).count_ones();
    }
    dist
}

// ============================================================================
// Slice implementations (for Q2: arrays vs slices)
// ============================================================================

/// Hamming distance on u8 slices (baseline slice API).
#[inline]
pub fn hamming_slice(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Hamming distance on u8 slices with assert for multiple-of-8 length.
#[inline]
pub fn hamming_slice_assert_multiple8(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    assert!(a.len() % 8 == 0, "length must be multiple of 8");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Hamming distance on u8 slices, processing as u64 chunks.
#[inline]
pub fn hamming_slice_u64_chunks(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    let mut dist = 0u32;
    for (a_chunk, b_chunk) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        let a_val = u64::from_ne_bytes(a_chunk.try_into().unwrap());
        let b_val = u64::from_ne_bytes(b_chunk.try_into().unwrap());
        dist += (a_val ^ b_val).count_ones();
    }
    // Handle remainder
    for (x, y) in a
        .chunks_exact(8)
        .remainder()
        .iter()
        .zip(b.chunks_exact(8).remainder())
    {
        dist += (x ^ y).count_ones();
    }
    dist
}

// ============================================================================
// Multiversion implementations (for Q3: SIMD dispatch)
// ============================================================================

#[cfg(feature = "multiversion_x86")]
#[multiversion::multiversion(targets(
    "x86_64+avx512vpopcntdq+avx512vl",
    "x86_64+avx512bw+avx512vl",
    "x86_64+avx2+popcnt",
    "x86_64+sse4.2+popcnt",
    "x86+avx2+popcnt",
    "x86+sse4.2+popcnt",
    "aarch64+neon",
))]
#[inline]
pub fn hamming_multiversion<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    a.chunks_exact(8)
        .zip(b.chunks_exact(8))
        .map(|(a_chunk, b_chunk)| {
            let a_val = u64::from_ne_bytes(a_chunk.try_into().unwrap());
            let b_val = u64::from_ne_bytes(b_chunk.try_into().unwrap());
            (a_val ^ b_val).count_ones()
        })
        .sum()
}

// ============================================================================
// Batch implementations (for Q4: batching)
// ============================================================================

/// Batch hamming with variable-size output slice.
pub fn hamming_batch_variable<const N: usize>(
    source: &[u8; N],
    targets: &[[u8; N]],
    out: &mut [u32],
) {
    assert_eq!(targets.len(), out.len());
    for (i, target) in targets.iter().enumerate() {
        out[i] = hamming_u8_iter(source, target);
    }
}

/// Batch hamming with fixed-size output array.
pub fn hamming_batch_fixed<const N: usize, const B: usize>(
    source: &[u8; N],
    targets: &[[u8; N]; B],
    out: &mut [u32; B],
) {
    for i in 0..B {
        out[i] = hamming_u8_iter(source, &targets[i]);
    }
}

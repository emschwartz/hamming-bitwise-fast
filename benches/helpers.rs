//! Shared helpers for divan benchmarks.
//!
//! This module provides test data generation and implementation variants
//! used across all benchmark files.

#![allow(dead_code)]

use rand::Rng;

// ============================================================================
// Type aliases
// ============================================================================

pub type Embedding<const N: usize> = [u64; N];
pub type EmbeddingBytes<const N: usize> = [u8; N];

// ============================================================================
// Test data generation
// ============================================================================

pub fn random_embedding<const N: usize>() -> Embedding<N> {
    let mut rng = rand::thread_rng();
    std::array::from_fn(|_| rng.gen())
}

pub fn random_embeddings<const N: usize>(count: usize) -> Vec<Embedding<N>> {
    (0..count).map(|_| random_embedding()).collect()
}

pub fn random_bytes<const N: usize>() -> [u8; N] {
    let mut rng = rand::thread_rng();
    std::array::from_fn(|_| rng.gen())
}

pub fn random_bytes_vec(size: usize) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen()).collect()
}

pub fn bytes_to_embedding<const N: usize>(bytes: &[u8]) -> Embedding<N> {
    assert_eq!(bytes.len(), N * 8);
    std::array::from_fn(|i| u64::from_ne_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap()))
}

pub fn embedding_to_bytes<const N: usize>(emb: &Embedding<N>) -> Vec<u8> {
    emb.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

// ============================================================================
// u8 array implementations (for Q1: data type comparison)
// ============================================================================

/// Hamming distance on u8 arrays using iterator.
#[inline]
pub fn hamming_u8_iter<const N: usize>(a: &EmbeddingBytes<N>, b: &EmbeddingBytes<N>) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Hamming distance on u8 arrays using for loop.
#[inline]
pub fn hamming_u8_for<const N: usize>(a: &EmbeddingBytes<N>, b: &EmbeddingBytes<N>) -> u32 {
    let mut dist = 0u32;
    for i in 0..N {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

/// Hamming distance taking u8 arrays but casting to u64 on x86.
/// N must be a multiple of 8 (enforced at compile time via const assertion).
/// This is the inverse of u64_as_u8 - convenient u8 API, but u64 performance on x86.
#[inline]
pub fn hamming_u8_as_u64<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    // Compile-time assertion that N is a multiple of 8
    const { assert!(N % 8 == 0, "N must be a multiple of 8") };

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // SAFETY: N is a multiple of 8 (enforced above), so [u8; N] can be read as [u64; N/8].
        // u8 arrays from random_bytes are stack-allocated and will be 8-byte aligned.
        let a_u64: &[u64] =
            unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u64, N / 8) };
        let b_u64: &[u64] =
            unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u64, N / 8) };
        a_u64
            .iter()
            .zip(b_u64.iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum()
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        // On non-x86, just use u8 directly
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum()
    }
}

/// Hamming distance on u8 arrays using chunks_exact(8) to hint vectorization.
/// Processes 8 bytes at a time as u64 without unsafe - lets the compiler know
/// we're stepping by 8 so it can potentially auto-vectorize better.
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

// ============================================================================
// u64 array implementations (for Q1: data type comparison)
// ============================================================================

/// Hamming distance on u64 arrays using iterator.
#[inline]
pub fn hamming_u64_iter<const N: usize>(a: &Embedding<N>, b: &Embedding<N>) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Hamming distance taking u64 arrays but casting to u8 on ARM.
/// This is the strategy the library uses - accept u64 for API convenience,
/// but process as u8 on ARM where NEON prefers byte operations.
#[inline]
pub fn hamming_u64_as_u8<const N: usize>(a: &Embedding<N>, b: &Embedding<N>) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: [u64; N] has same layout as [u8; N*8], u8 has alignment 1
        let a_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, N * 8) };
        let b_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u8, N * 8) };
        a_bytes
            .iter()
            .zip(b_bytes.iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum()
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // On non-ARM, just use u64 directly
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum()
    }
}

/// Hamming distance on u64 arrays using for loop.
#[inline]
pub fn hamming_u64_for<const N: usize>(a: &Embedding<N>, b: &Embedding<N>) -> u32 {
    let mut dist = 0u32;
    for i in 0..N {
        dist += (a[i] ^ b[i]).count_ones();
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
/// This hint may help the compiler optimize better.
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
    for (x, y) in a.chunks_exact(8).remainder().iter().zip(b.chunks_exact(8).remainder()) {
        dist += (x ^ y).count_ones();
    }
    dist
}

// ============================================================================
// Multiversion implementations (for Q3: SIMD dispatch)
// ============================================================================

#[cfg(feature = "multiversion")]
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
pub fn hamming_multiversion<const N: usize>(a: &Embedding<N>, b: &Embedding<N>) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

// ============================================================================
// Pulp implementations (for Q3: SIMD dispatch)
// ============================================================================

/// Hamming distance using pulp's portable SIMD dispatch.
pub fn hamming_pulp<const N: usize>(a: &Embedding<N>, b: &Embedding<N>) -> u32 {
    use pulp::Simd;

    struct HammingOp<'a, const N: usize> {
        a: &'a Embedding<N>,
        b: &'a Embedding<N>,
    }

    impl<const N: usize> pulp::WithSimd for HammingOp<'_, N> {
        type Output = u32;

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let _ = simd;
            self.a
                .iter()
                .zip(self.b.iter())
                .map(|(x, y)| (x ^ y).count_ones())
                .sum()
        }
    }

    pulp::Arch::new().dispatch(HammingOp { a, b })
}

// ============================================================================
// Batch implementations (for Q4: batching)
// ============================================================================

/// Batch hamming with variable-size output slice.
pub fn hamming_batch_variable<const N: usize>(
    source: &Embedding<N>,
    targets: &[Embedding<N>],
    out: &mut [u32],
) {
    assert_eq!(targets.len(), out.len());
    for (i, target) in targets.iter().enumerate() {
        out[i] = hamming_u64_iter(source, target);
    }
}

/// Batch hamming with fixed-size output array.
pub fn hamming_batch_fixed<const N: usize, const B: usize>(
    source: &Embedding<N>,
    targets: &[Embedding<N>; B],
    out: &mut [u32; B],
) {
    for i in 0..B {
        out[i] = hamming_u64_iter(source, &targets[i]);
    }
}

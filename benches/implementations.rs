//! Benchmark-only implementations for comparing different approaches.
//!
//! These implementations are NOT part of the public API. They exist solely
//! for benchmarking to understand performance characteristics of different
//! techniques (loop styles, data representations, dispatch strategies).

#![allow(dead_code)]

use rand::Rng;

// ============================================================================
// Type aliases for benchmark convenience
// ============================================================================

/// A fixed-size embedding represented as an array of N u64 values.
pub type Embedding<const N: usize> = [u64; N];

/// A fixed-size embedding represented as an array of N bytes (u8).
pub type EmbeddingBytes<const N: usize> = [u8; N];

// ============================================================================
// Test data generation helpers
// ============================================================================

pub fn random_embedding<const N: usize>() -> Embedding<N> {
    let mut rng = rand::thread_rng();
    let mut emb = [0u64; N];
    for i in 0..N {
        emb[i] = rng.gen();
    }
    emb
}

pub fn random_embeddings<const N: usize>(count: usize) -> Vec<Embedding<N>> {
    (0..count).map(|_| random_embedding()).collect()
}

pub fn random_bytes<const N: usize>() -> [u8; N] {
    let mut rng = rand::thread_rng();
    let mut arr = [0u8; N];
    for i in 0..N {
        arr[i] = rng.gen();
    }
    arr
}

pub fn random_bytes_vec(size: usize) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen()).collect()
}

// ============================================================================
// Conversion helpers
// ============================================================================

/// Convert a byte slice to a fixed-size embedding.
pub fn bytes_to_embedding<const N: usize>(bytes: &[u8]) -> Embedding<N> {
    assert_eq!(
        bytes.len(),
        N * 8,
        "Byte slice must be exactly {} bytes",
        N * 8
    );
    let mut result = [0u64; N];
    for (i, chunk) in bytes.chunks_exact(8).enumerate() {
        result[i] = u64::from_ne_bytes(chunk.try_into().unwrap());
    }
    result
}

// ============================================================================
// Naive implementations (for correctness testing)
// ============================================================================

/// Naive hamming distance using a for loop.
#[inline]
pub fn naive_hamming_distance(x: &[u8], y: &[u8]) -> u64 {
    assert_eq!(x.len(), y.len());
    let mut distance: u32 = 0;
    for i in 0..x.len() {
        distance += (x[i] ^ y[i]).count_ones();
    }
    distance as u64
}

/// Naive hamming distance using iterators.
#[inline]
pub fn naive_hamming_distance_iter(x: &[u8], y: &[u8]) -> u64 {
    x.iter()
        .zip(y)
        .fold(0, |a, (b, c)| a + (*b ^ *c).count_ones()) as u64
}

// ============================================================================
// u64 array implementations (loop style variants)
// ============================================================================

/// Hamming distance using reference parameters and a for loop.
#[inline]
pub fn hamming_ref_for<const N: usize>(a: &Embedding<N>, b: &Embedding<N>) -> u32 {
    let mut dist = 0u32;
    for i in 0..N {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

/// Hamming distance using reference parameters and an iterator chain.
#[inline]
pub fn hamming_ref_iter<const N: usize>(a: &Embedding<N>, b: &Embedding<N>) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Hamming distance using stack-copied parameters and a for loop.
#[inline]
pub fn hamming_copy_for<const N: usize>(a: Embedding<N>, b: Embedding<N>) -> u32 {
    let mut dist = 0u32;
    for i in 0..N {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

/// Hamming distance using stack-copied parameters and an iterator chain.
#[inline]
pub fn hamming_copy_iter<const N: usize>(a: Embedding<N>, b: Embedding<N>) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

// ============================================================================
// u8 array implementations (for representation comparison)
// ============================================================================

/// Hamming distance on u8 arrays using a for loop.
#[inline]
pub fn hamming_u8_for<const N: usize>(a: &EmbeddingBytes<N>, b: &EmbeddingBytes<N>) -> u32 {
    let mut dist = 0u32;
    for i in 0..N {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

/// Hamming distance on u8 arrays using an iterator.
#[inline]
pub fn hamming_u8_iter<const N: usize>(a: &EmbeddingBytes<N>, b: &EmbeddingBytes<N>) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Hamming distance on u8 slices with assert for alignment.
#[inline]
pub fn hamming_slice_assert_aligned(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    assert!(a.len() % 16 == 0, "length must be multiple of 16");

    let mut dist = 0u32;
    for i in 0..a.len() {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

/// Hamming distance on u8 slices processing as u64 chunks with assert.
#[inline]
pub fn hamming_slice_assert_u64_chunks(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    assert!(a.len() % 8 == 0, "length must be multiple of 8");

    let mut dist = 0u32;
    for (a_chunk, b_chunk) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        let a_val = u64::from_ne_bytes(a_chunk.try_into().unwrap());
        let b_val = u64::from_ne_bytes(b_chunk.try_into().unwrap());
        dist += (a_val ^ b_val).count_ones();
    }
    dist
}

/// Hamming distance on u8 arrays, processing 8 bytes at a time as u64.
#[inline]
pub fn hamming_u8_chunked<const N: usize>(a: &EmbeddingBytes<N>, b: &EmbeddingBytes<N>) -> u32 {
    let mut dist = 0u32;
    let chunks = N / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let a_chunk = u64::from_ne_bytes([
            a[offset],
            a[offset + 1],
            a[offset + 2],
            a[offset + 3],
            a[offset + 4],
            a[offset + 5],
            a[offset + 6],
            a[offset + 7],
        ]);
        let b_chunk = u64::from_ne_bytes([
            b[offset],
            b[offset + 1],
            b[offset + 2],
            b[offset + 3],
            b[offset + 4],
            b[offset + 5],
            b[offset + 6],
            b[offset + 7],
        ]);
        dist += (a_chunk ^ b_chunk).count_ones();
    }

    // Handle remainder
    let remainder_start = chunks * 8;
    for i in remainder_start..N {
        dist += (a[i] ^ b[i]).count_ones();
    }

    dist
}

// ============================================================================
// Batch operations (non-multiversion variants for benchmarking)
// ============================================================================

/// Non-multiversion batch operation for baseline comparison.
pub fn hamming_batch_into_auto<const N: usize>(
    source: &Embedding<N>,
    targets: &[Embedding<N>],
    out: &mut [u32],
) {
    assert_eq!(targets.len(), out.len());
    for (i, target) in targets.iter().enumerate() {
        let mut dist = 0u32;
        for j in 0..N {
            dist += (source[j] ^ target[j]).count_ones();
        }
        out[i] = dist;
    }
}

/// Fixed-size batch operation for testing unrolling hypothesis.
pub fn hamming_batch_fixed_auto<const N: usize, const B: usize>(
    source: &Embedding<N>,
    targets: &[Embedding<N>; B],
    out: &mut [u32; B],
) {
    for i in 0..B {
        let mut dist = 0u32;
        for j in 0..N {
            dist += (source[j] ^ targets[i][j]).count_ones();
        }
        out[i] = dist;
    }
}

// ============================================================================
// Multiversion implementations (dispatch strategy comparison)
// ============================================================================

/// Hamming distance with runtime CPU feature detection.
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

/// Batch with multiversion dispatch.
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
pub fn hamming_batch_into<const N: usize>(
    source: &Embedding<N>,
    targets: &[Embedding<N>],
    out: &mut [u32],
) {
    assert_eq!(targets.len(), out.len());
    for (i, target) in targets.iter().enumerate() {
        let mut dist = 0u32;
        for j in 0..N {
            dist += (source[j] ^ target[j]).count_ones();
        }
        out[i] = dist;
    }
}

/// Fixed-size batch with multiversion dispatch.
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
pub fn hamming_batch_fixed<const N: usize, const B: usize>(
    source: &Embedding<N>,
    targets: &[Embedding<N>; B],
    out: &mut [u32; B],
) {
    for i in 0..B {
        let mut dist = 0u32;
        for j in 0..N {
            dist += (source[j] ^ targets[i][j]).count_ones();
        }
        out[i] = dist;
    }
}

// ============================================================================
// pulp implementations (portable SIMD comparison)
// ============================================================================

/// Hamming distance using portable SIMD via the `pulp` crate.
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

/// Batch hamming distance using pulp for portable SIMD.
#[allow(dead_code)]
pub fn hamming_batch_pulp<const N: usize>(
    source: &Embedding<N>,
    targets: &[Embedding<N>],
    out: &mut [u32],
) {
    use pulp::Simd;

    struct BatchHammingOp<'a, const N: usize> {
        source: &'a Embedding<N>,
        targets: &'a [Embedding<N>],
        out: &'a mut [u32],
    }

    impl<const N: usize> pulp::WithSimd for BatchHammingOp<'_, N> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let _ = simd;
            assert_eq!(self.targets.len(), self.out.len());
            for (i, target) in self.targets.iter().enumerate() {
                let mut dist = 0u32;
                for j in 0..N {
                    dist += (self.source[j] ^ target[j]).count_ones();
                }
                self.out[i] = dist;
            }
        }
    }

    pulp::Arch::new().dispatch(BatchHammingOp { source, targets, out })
}

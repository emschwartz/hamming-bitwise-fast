//! A fast, zero-dependency implementation of bitwise Hamming Distance using
//! a method amenable to auto-vectorization.

/// Calculate the bitwise Hamming distance between two byte slices.
///
/// While this implementation does not explicitly use SIMD, it uses
/// a technique that is amenable to auto-vectorization. Its performance
/// is similar to or faster than more complex implementations that use
/// explicit SIMD instructions for specific architectures.
///
/// # Panics
///
/// Panics if the two slices are not the same length.
#[inline]
pub fn hamming_bitwise_fast(x: &[u8], y: &[u8]) -> u32 {
    assert_eq!(x.len(), y.len());

    // Process 8 bytes at a time using u64
    let mut distance = x
        .chunks_exact(8)
        .zip(y.chunks_exact(8))
        .map(|(x_chunk, y_chunk)| {
            // This is safe because we know the chunks are exactly 8 bytes.
            // Also, we don't care whether the platform uses little-endian or big-endian
            // byte order. Since we're only XORing values, we just care that the
            // endianness is the same for both.
            let x_val = u64::from_ne_bytes(x_chunk.try_into().unwrap());
            let y_val = u64::from_ne_bytes(y_chunk.try_into().unwrap());
            (x_val ^ y_val).count_ones()
        })
        .sum::<u32>();

    if x.len() % 8 != 0 {
        distance += x
            .chunks_exact(8)
            .remainder()
            .iter()
            .zip(y.chunks_exact(8).remainder())
            .map(|(x_byte, y_byte)| (x_byte ^ y_byte).count_ones())
            .sum::<u32>();
    }

    distance
}

#[doc(hidden)]
#[inline]
pub fn naive_hamming_distance(x: &[u8], y: &[u8]) -> u64 {
    assert_eq!(x.len(), y.len());
    let mut distance: u32 = 0;
    for i in 0..x.len() {
        distance += (x[i] ^ y[i]).count_ones();
    }
    distance as u64
}

#[doc(hidden)]
#[inline]
pub fn naive_hamming_distance_iter(x: &[u8], y: &[u8]) -> u64 {
    x.iter()
        .zip(y)
        .fold(0, |a, (b, c)| a + (*b ^ *c).count_ones()) as u64
}

// ============================================================================
// Const-generic implementations for fixed-size embeddings
// ============================================================================

/// A fixed-size embedding represented as an array of N u64 values.
///
/// Common sizes:
/// - N=8: 512-bit embedding
/// - N=12: 768-bit embedding
/// - N=16: 1024-bit embedding
/// - N=32: 2048-bit embedding
pub type Embedding<const N: usize> = [u64; N];

/// Hamming distance using reference parameters and a for loop.
///
/// This is the simplest const-generic implementation. The compiler knows
/// the exact size at compile time, enabling loop unrolling.
#[inline]
pub fn hamming_ref_for<const N: usize>(a: &Embedding<N>, b: &Embedding<N>) -> u32 {
    let mut dist = 0u32;
    for i in 0..N {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

/// Hamming distance using reference parameters and an iterator chain.
///
/// Functionally equivalent to `hamming_ref_for`, but uses iterators
/// which may help the compiler recognize vectorization patterns.
#[inline]
pub fn hamming_ref_iter<const N: usize>(a: &Embedding<N>, b: &Embedding<N>) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Hamming distance using stack-copied parameters and a for loop.
///
/// For small N (≤16), copying to the stack may be faster because
/// the values can live entirely in registers, avoiding pointer indirection.
#[inline]
pub fn hamming_copy_for<const N: usize>(a: Embedding<N>, b: Embedding<N>) -> u32 {
    let mut dist = 0u32;
    for i in 0..N {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

/// Hamming distance using stack-copied parameters and an iterator chain.
///
/// Combines stack copying with iterator-based computation.
#[inline]
pub fn hamming_copy_iter<const N: usize>(a: Embedding<N>, b: Embedding<N>) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

// ============================================================================
// u8-based implementations for comparison
// ============================================================================

/// A fixed-size embedding represented as an array of N bytes (u8).
///
/// For comparing against u64-based representations.
/// Common sizes:
/// - N=64: 512-bit embedding
/// - N=96: 768-bit embedding
/// - N=128: 1024-bit embedding
/// - N=256: 2048-bit embedding
pub type EmbeddingBytes<const N: usize> = [u8; N];

/// Hamming distance on u8 arrays using a for loop.
///
/// Processes one byte at a time - less efficient than u64.
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
///
/// Tests whether asserting the length is a multiple of 8/16/etc
/// helps the compiler vectorize.
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
    // Process 8 bytes at a time
    for (a_chunk, b_chunk) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        let a_val = u64::from_ne_bytes(a_chunk.try_into().unwrap());
        let b_val = u64::from_ne_bytes(b_chunk.try_into().unwrap());
        dist += (a_val ^ b_val).count_ones();
    }
    dist
}

/// Hamming distance on u8 arrays, processing 8 bytes at a time as u64.
///
/// This is essentially what `hamming_bitwise_fast` does, but with const generics.
#[inline]
pub fn hamming_u8_chunked<const N: usize>(a: &EmbeddingBytes<N>, b: &EmbeddingBytes<N>) -> u32 {
    let mut dist = 0u32;

    // Process 8 bytes at a time
    let chunks = N / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let a_chunk = u64::from_ne_bytes([
            a[offset], a[offset + 1], a[offset + 2], a[offset + 3],
            a[offset + 4], a[offset + 5], a[offset + 6], a[offset + 7],
        ]);
        let b_chunk = u64::from_ne_bytes([
            b[offset], b[offset + 1], b[offset + 2], b[offset + 3],
            b[offset + 4], b[offset + 5], b[offset + 6], b[offset + 7],
        ]);
        dist += (a_chunk ^ b_chunk).count_ones();
    }

    // Handle remainder (if N is not divisible by 8)
    let remainder_start = chunks * 8;
    for i in remainder_start..N {
        dist += (a[i] ^ b[i]).count_ones();
    }

    dist
}

// ============================================================================
// Multiversion implementations with runtime CPU dispatch
// ============================================================================

/// Hamming distance with runtime CPU feature detection.
///
/// Uses the `multiversion` crate to dispatch to the optimal implementation
/// based on the CPU's capabilities at runtime. This adds a small dispatch
/// overhead per call, but ensures optimal performance across different CPUs.
///
/// Supported targets (in priority order):
/// - x86_64: AVX-512 VPOPCNTDQ (Ice Lake+, Zen 4+), AVX-512BW, AVX2+POPCNT, SSE4.2+POPCNT
/// - x86 (32-bit): AVX2+POPCNT, SSE4.2+POPCNT
/// - aarch64: NEON (Apple Silicon, Graviton, all AArch64)
#[cfg(feature = "multiversion")]
#[multiversion::multiversion(targets(
    "x86_64+avx512vpopcntdq+avx512vl",  // Intel Ice Lake+, AMD Zen 4+
    "x86_64+avx512bw+avx512vl",          // Intel Skylake-X, AMD Zen 4 (no VPOPCNTDQ)
    "x86_64+avx2+popcnt",                // Intel Haswell+, AMD Zen 1-3
    "x86_64+sse4.2+popcnt",              // Intel Nehalem+, AMD K10+
    "x86+avx2+popcnt",                   // 32-bit x86 with AVX2
    "x86+sse4.2+popcnt",                 // 32-bit x86 fallback
    "aarch64+neon",                      // Apple Silicon, Graviton, all AArch64
))]
#[inline]
pub fn hamming_multiversion<const N: usize>(a: &Embedding<N>, b: &Embedding<N>) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

// ============================================================================
// Batch operations - single dispatch for multiple comparisons
// ============================================================================

/// Compute hamming distance from one source to many targets.
///
/// This function amortizes the multiversion dispatch overhead across
/// all comparisons. The source embedding stays in registers while
/// iterating through all targets.
///
/// # Arguments
/// * `source` - The source embedding to compare against
/// * `targets` - Slice of target embeddings
/// * `out` - Output buffer for distances (must be same length as targets)
///
/// # Panics
/// Panics if `out.len() != targets.len()`
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

/// Non-multiversion batch operation for baseline comparison.
///
/// Same as `hamming_batch_into` but without multiversion dispatch,
/// relying purely on auto-vectorization.
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
///
/// Uses const generics for both embedding size and batch size,
/// allowing the compiler to potentially unroll both loops.
///
/// Hypothesis: This might be slower due to batch management overhead.
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

/// Non-multiversion fixed-size batch for baseline comparison.
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
// Helper functions for converting between representations
// ============================================================================

/// Convert a byte slice to a fixed-size embedding.
///
/// # Panics
/// Panics if the slice length doesn't match N * 8 bytes.
pub fn bytes_to_embedding<const N: usize>(bytes: &[u8]) -> Embedding<N> {
    assert_eq!(bytes.len(), N * 8, "Byte slice must be exactly {} bytes", N * 8);
    let mut result = [0u64; N];
    for (i, chunk) in bytes.chunks_exact(8).enumerate() {
        result[i] = u64::from_ne_bytes(chunk.try_into().unwrap());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_same_results() {
        let a_hex = "cd8e98b29187133982909fc8b30e39c7b4dca73128ece9cf22ce64eefcf75a3adb0f129b1b00f63a20209e83cb873df707f1af6a4e3558941556b215461a9cbbbce984233c8b8a51e8bd2d1e7f6500caf59fb497440d15365b81e75d3ca4fc9947d5fcb97a0a7b5e44a6b93ee4f622c9b3157991fecac58f364b23f01fd8621e";
        let b_hex = "860e297e5ce51d3bee094b69bedaaf4ec5d74aa639fec1980ac8d6debb77ff8a323350ab4217867a2521d1248f878dc71f39ede3ea357ef39065da261f9ab470ce6884a3e8a6727d1a3c2614ab66481683f63c01de17b4f59d11659ab5a4310121fccc69418839ff6783f9ce7d760ac8e3db7824eef28d0f12fc6b3c1ef8d75c";
        let a_bytes = hex::decode(a_hex).unwrap();
        let b_bytes = hex::decode(b_hex).unwrap();

        let expected = naive_hamming_distance(&a_bytes, &b_bytes);

        // Compare with naive_iter implementation
        assert_eq!(expected, naive_hamming_distance_iter(&a_bytes, &b_bytes));

        // Compare with auto vectorized implementation
        assert_eq!(expected, hamming_bitwise_fast(&a_bytes, &b_bytes) as u64);

        // Compare with hamming crate
        assert_eq!(expected, hamming::distance_fast(&a_bytes, &b_bytes).unwrap());

        // Compare with hamming_rs crate (x86/x86_64 only)
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        assert_eq!(expected, hamming_rs::distance_faster(&a_bytes, &b_bytes));

        // Compare with simsimd crate
        assert_eq!(
            expected,
            simsimd::BinarySimilarity::hamming(&a_bytes, &b_bytes).unwrap() as u64
        );
    }

    #[test]
    fn const_generic_implementations_match() {
        let a_hex = "cd8e98b29187133982909fc8b30e39c7b4dca73128ece9cf22ce64eefcf75a3adb0f129b1b00f63a20209e83cb873df707f1af6a4e3558941556b215461a9cbbbce984233c8b8a51e8bd2d1e7f6500caf59fb497440d15365b81e75d3ca4fc9947d5fcb97a0a7b5e44a6b93ee4f622c9b3157991fecac58f364b23f01fd8621e";
        let b_hex = "860e297e5ce51d3bee094b69bedaaf4ec5d74aa639fec1980ac8d6debb77ff8a323350ab4217867a2521d1248f878dc71f39ede3ea357ef39065da261f9ab470ce6884a3e8a6727d1a3c2614ab66481683f63c01de17b4f59d11659ab5a4310121fccc69418839ff6783f9ce7d760ac8e3db7824eef28d0f12fc6b3c1ef8d75c";
        let a_bytes = hex::decode(a_hex).unwrap();
        let b_bytes = hex::decode(b_hex).unwrap();

        // Convert to fixed-size embeddings (128 bytes = 16 u64s = 1024 bits)
        let a: Embedding<16> = bytes_to_embedding(&a_bytes);
        let b: Embedding<16> = bytes_to_embedding(&b_bytes);

        let expected = naive_hamming_distance(&a_bytes, &b_bytes) as u32;

        // Test all const-generic implementations
        assert_eq!(expected, hamming_ref_for(&a, &b), "hamming_ref_for mismatch");
        assert_eq!(expected, hamming_ref_iter(&a, &b), "hamming_ref_iter mismatch");
        assert_eq!(expected, hamming_copy_for(a, b), "hamming_copy_for mismatch");
        assert_eq!(expected, hamming_copy_iter(a, b), "hamming_copy_iter mismatch");

        // Test multiversion implementation if feature is enabled
        #[cfg(feature = "multiversion")]
        assert_eq!(expected, hamming_multiversion(&a, &b), "hamming_multiversion mismatch");
    }

    #[test]
    fn batch_operations_match() {
        let a_hex = "cd8e98b29187133982909fc8b30e39c7b4dca73128ece9cf22ce64eefcf75a3adb0f129b1b00f63a20209e83cb873df707f1af6a4e3558941556b215461a9cbbbce984233c8b8a51e8bd2d1e7f6500caf59fb497440d15365b81e75d3ca4fc9947d5fcb97a0a7b5e44a6b93ee4f622c9b3157991fecac58f364b23f01fd8621e";
        let b_hex = "860e297e5ce51d3bee094b69bedaaf4ec5d74aa639fec1980ac8d6debb77ff8a323350ab4217867a2521d1248f878dc71f39ede3ea357ef39065da261f9ab470ce6884a3e8a6727d1a3c2614ab66481683f63c01de17b4f59d11659ab5a4310121fccc69418839ff6783f9ce7d760ac8e3db7824eef28d0f12fc6b3c1ef8d75c";
        let a_bytes = hex::decode(a_hex).unwrap();
        let b_bytes = hex::decode(b_hex).unwrap();

        let source: Embedding<16> = bytes_to_embedding(&a_bytes);
        let target: Embedding<16> = bytes_to_embedding(&b_bytes);
        let targets = [target; 10];

        let expected = hamming_ref_for(&source, &target);

        // Test batch_into_auto
        let mut out = [0u32; 10];
        hamming_batch_into_auto(&source, &targets, &mut out);
        for (i, &dist) in out.iter().enumerate() {
            assert_eq!(expected, dist, "hamming_batch_into_auto mismatch at index {}", i);
        }

        // Test batch_fixed_auto
        let mut out_fixed = [0u32; 10];
        hamming_batch_fixed_auto(&source, &targets, &mut out_fixed);
        for (i, &dist) in out_fixed.iter().enumerate() {
            assert_eq!(expected, dist, "hamming_batch_fixed_auto mismatch at index {}", i);
        }

        #[cfg(feature = "multiversion")]
        {
            // Test batch_into with multiversion
            let mut out_mv = [0u32; 10];
            hamming_batch_into(&source, &targets, &mut out_mv);
            for (i, &dist) in out_mv.iter().enumerate() {
                assert_eq!(expected, dist, "hamming_batch_into mismatch at index {}", i);
            }

            // Test batch_fixed with multiversion
            let mut out_fixed_mv = [0u32; 10];
            hamming_batch_fixed(&source, &targets, &mut out_fixed_mv);
            for (i, &dist) in out_fixed_mv.iter().enumerate() {
                assert_eq!(expected, dist, "hamming_batch_fixed mismatch at index {}", i);
            }
        }
    }

    #[test]
    fn bytes_to_embedding_roundtrip() {
        let bytes: Vec<u8> = (0..128).collect();
        let emb: Embedding<16> = bytes_to_embedding(&bytes);

        // Verify the conversion preserved the data
        for (i, chunk) in bytes.chunks_exact(8).enumerate() {
            let expected = u64::from_ne_bytes(chunk.try_into().unwrap());
            assert_eq!(expected, emb[i], "Mismatch at index {}", i);
        }
    }
}

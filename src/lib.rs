//! A fast, zero-dependency implementation of bitwise Hamming Distance using
//! a method amenable to auto-vectorization.
//!
//! # Quick Start
//!
//! For byte slices (variable-length):
//! ```
//! use hamming_bitwise_fast::hamming_bitwise_fast;
//!
//! let a = [0u8; 128];
//! let b = [0xFFu8; 128];
//! let distance = hamming_bitwise_fast(&a, &b);
//! ```
//!
//! For fixed-size embeddings (faster, recommended for known sizes):
//! ```
//! use hamming_bitwise_fast::hamming;
//!
//! // 1024-bit embeddings = 128 bytes
//! let a: [u8; 128] = [0; 128];
//! let b: [u8; 128] = [0xFF; 128];
//! let distance = hamming(&a, &b);
//! ```
//!
//! For batch operations:
//! ```
//! use hamming_bitwise_fast::hamming_batch;
//!
//! let source: [u8; 128] = [0; 128];
//! let targets: Vec<[u8; 128]> = vec![[1; 128], [2; 128], [3; 128]];
//! let mut distances = vec![0u32; targets.len()];
//! hamming_batch(&source, &targets, &mut distances);
//! ```
//!
//! # Feature Flags
//!
//! - `multiversion`: Enables runtime CPU dispatch for optimal SIMD on x86.
//!   Recommended for x86 deployments. On ARM, auto-vectorization is already
//!   near-optimal, so this adds minimal benefit.

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
            let x_val = u64::from_ne_bytes(x_chunk.try_into().unwrap());
            let y_val = u64::from_ne_bytes(y_chunk.try_into().unwrap());
            (x_val ^ y_val).count_ones()
        })
        .sum::<u32>();

    // Handle remainder bytes
    for (x_byte, y_byte) in x
        .chunks_exact(8)
        .remainder()
        .iter()
        .zip(y.chunks_exact(8).remainder())
    {
        distance += (x_byte ^ y_byte).count_ones();
    }

    distance
}

// ============================================================================
// Platform-optimized const-generic implementations
// ============================================================================

/// Compute Hamming distance for fixed-size byte arrays.
///
/// This is the recommended function for embeddings of known size at compile time.
/// The const generic `N` represents the number of bytes.
///
/// Common sizes:
/// - `N=64`: 512-bit embedding
/// - `N=96`: 768-bit embedding
/// - `N=128`: 1024-bit embedding
/// - `N=256`: 2048-bit embedding
///
/// # Performance
///
/// - On ARM (M2, Graviton): Uses NEON byte-level operations (~2ns for 1024-bit)
/// - On x86 with `multiversion`: Uses AVX-512 VPOPCNTDQ when available (~1-2ns)
/// - On x86 without features: Uses chunked u64 processing (~8ns for 1024-bit)
///
/// # Example
///
/// ```
/// use hamming_bitwise_fast::hamming;
///
/// // 1024-bit embeddings = 128 bytes
/// let a: [u8; 128] = [0x12; 128];
/// let b: [u8; 128] = [0xFE; 128];
/// let distance = hamming(&a, &b);
/// ```
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
pub fn hamming<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    hamming_inner(a, b)
}

/// Compute Hamming distance for fixed-size byte arrays (non-multiversion).
#[cfg(not(feature = "multiversion"))]
#[inline]
pub fn hamming<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    hamming_inner(a, b)
}

/// Internal implementation that selects the optimal strategy per platform.
#[inline]
fn hamming_inner<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    // On x86: use chunks_exact(8) to process as u64 (enables AVX-512 VPOPCNTDQ)
    // The remainder handling is written so the compiler can optimize it away
    // when N is a compile-time constant that's a multiple of 8.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let chunks_distance: u32 = a
            .chunks_exact(8)
            .zip(b.chunks_exact(8))
            .map(|(a_chunk, b_chunk)| {
                let a_val = u64::from_ne_bytes(a_chunk.try_into().unwrap());
                let b_val = u64::from_ne_bytes(b_chunk.try_into().unwrap());
                (a_val ^ b_val).count_ones()
            })
            .sum();

        // Handle remainder bytes by packing into a u64 for a single popcount.
        // Compiler optimizes this away when N % 8 == 0.
        let remainder_distance = if N % 8 != 0 {
            let rem_start = (N / 8) * 8;
            let a_rem = &a[rem_start..];
            let b_rem = &b[rem_start..];
            let mut a_val = 0u64;
            let mut b_val = 0u64;
            for (i, (&a_byte, &b_byte)) in a_rem.iter().zip(b_rem).enumerate() {
                a_val |= (a_byte as u64) << (i * 8);
                b_val |= (b_byte as u64) << (i * 8);
            }
            (a_val ^ b_val).count_ones()
        } else {
            0
        };

        chunks_distance + remainder_distance
    }

    // On ARM and other architectures: byte-by-byte iteration
    // (NEON handles this efficiently, and it's a safe default for unknown archs)
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum()
    }
}

/// Compute Hamming distance from one source embedding to many targets.
///
/// This is significantly faster than calling [`hamming`] in a loop because:
/// 1. The function call overhead is amortized across all comparisons
/// 2. The source embedding can stay in registers
/// 3. With `multiversion`, the CPU dispatch happens once for all comparisons
///
/// # Arguments
///
/// * `source` - The source embedding to compare against all targets
/// * `targets` - Slice of target embeddings
/// * `out` - Output buffer for distances (must have same length as `targets`)
///
/// # Panics
///
/// Panics if `out.len() != targets.len()`
///
/// # Example
///
/// ```
/// use hamming_bitwise_fast::hamming_batch;
///
/// let source: [u8; 128] = [0; 128];
/// let targets = vec![[1u8; 128], [2u8; 128], [3u8; 128]];
/// let mut distances = vec![0u32; 3];
///
/// hamming_batch(&source, &targets, &mut distances);
/// ```
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
pub fn hamming_batch<const N: usize>(source: &[u8; N], targets: &[[u8; N]], out: &mut [u32]) {
    hamming_batch_inner(source, targets, out)
}

/// Compute Hamming distance from one source to many targets (non-multiversion).
#[cfg(not(feature = "multiversion"))]
pub fn hamming_batch<const N: usize>(source: &[u8; N], targets: &[[u8; N]], out: &mut [u32]) {
    hamming_batch_inner(source, targets, out)
}

/// Internal batch implementation.
#[inline]
fn hamming_batch_inner<const N: usize>(source: &[u8; N], targets: &[[u8; N]], out: &mut [u32]) {
    assert_eq!(targets.len(), out.len());

    // On x86: use chunks_exact(8) to process as u64 (enables AVX-512 VPOPCNTDQ)
    // The remainder handling is written so the compiler can optimize it away
    // when N is a compile-time constant that's a multiple of 8.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        targets.iter().zip(out.iter_mut()).for_each(|(target, dist)| {
            let chunks_distance: u32 = source
                .chunks_exact(8)
                .zip(target.chunks_exact(8))
                .map(|(a_chunk, b_chunk)| {
                    let a_val = u64::from_ne_bytes(a_chunk.try_into().unwrap());
                    let b_val = u64::from_ne_bytes(b_chunk.try_into().unwrap());
                    (a_val ^ b_val).count_ones()
                })
                .sum();

            // Handle remainder bytes by packing into a u64 for a single popcount.
            // Compiler optimizes this away when N % 8 == 0.
            let remainder_distance = if N % 8 != 0 {
                let rem_start = (N / 8) * 8;
                let source_rem = &source[rem_start..];
                let target_rem = &target[rem_start..];
                let mut a_val = 0u64;
                let mut b_val = 0u64;
                for (i, (&a_byte, &b_byte)) in source_rem.iter().zip(target_rem).enumerate() {
                    a_val |= (a_byte as u64) << (i * 8);
                    b_val |= (b_byte as u64) << (i * 8);
                }
                (a_val ^ b_val).count_ones()
            } else {
                0
            };

            *dist = chunks_distance + remainder_distance;
        });
    }

    // On ARM and other architectures: byte-by-byte iteration
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        targets.iter().zip(out.iter_mut()).for_each(|(target, dist)| {
            *dist = source
                .iter()
                .zip(target.iter())
                .map(|(x, y)| (x ^ y).count_ones())
                .sum();
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamming_bitwise_fast_correctness() {
        let a = [0u8; 128];
        let b = [0xFFu8; 128];
        assert_eq!(hamming_bitwise_fast(&a, &b), 1024);
        assert_eq!(hamming_bitwise_fast(&a, &a), 0);

        let mut c = [0u8; 128];
        c[0] = 1;
        assert_eq!(hamming_bitwise_fast(&a, &c), 1);
    }

    #[test]
    fn hamming_const_generic_correctness() {
        let a: [u8; 128] = [0; 128];
        let b: [u8; 128] = [0xFF; 128];
        assert_eq!(hamming(&a, &b), 1024);
        assert_eq!(hamming(&a, &a), 0);

        let mut c = [0u8; 128];
        c[0] = 1;
        assert_eq!(hamming(&a, &c), 1);
    }

    #[test]
    fn hamming_batch_correctness() {
        let source: [u8; 128] = [0; 128];
        let targets = vec![
            [0xFFu8; 128], // 1024 bits different
            [0u8; 128],    // 0 bits different
            [1u8; 128],    // 128 bits different (one bit per byte)
        ];
        let mut out = vec![0u32; 3];

        hamming_batch(&source, &targets, &mut out);

        assert_eq!(out[0], 1024);
        assert_eq!(out[1], 0);
        assert_eq!(out[2], 128);
    }

    #[test]
    fn hamming_matches_bitwise_fast() {
        let a: [u8; 128] = std::array::from_fn(|i| i as u8);
        let b: [u8; 128] = std::array::from_fn(|i| (i + 128) as u8);

        assert_eq!(hamming(&a, &b), hamming_bitwise_fast(&a, &b));
    }

    #[test]
    fn different_embedding_sizes() {
        // 512-bit (64 bytes)
        let a: [u8; 64] = [0; 64];
        let b: [u8; 64] = [0xFF; 64];
        assert_eq!(hamming(&a, &b), 512);

        // 768-bit (96 bytes)
        let a: [u8; 96] = [0; 96];
        let b: [u8; 96] = [0xFF; 96];
        assert_eq!(hamming(&a, &b), 768);

        // 2048-bit (256 bytes)
        let a: [u8; 256] = [0; 256];
        let b: [u8; 256] = [0xFF; 256];
        assert_eq!(hamming(&a, &b), 2048);
    }

    #[test]
    fn odd_sizes_with_remainder() {
        // 7 bytes (not a multiple of 8) - tests remainder handling
        let a: [u8; 7] = [0; 7];
        let b: [u8; 7] = [0xFF; 7];
        assert_eq!(hamming(&a, &b), 56); // 7 * 8 = 56 bits

        // 13 bytes (8 + 5 remainder)
        let a: [u8; 13] = [0; 13];
        let b: [u8; 13] = [0xFF; 13];
        assert_eq!(hamming(&a, &b), 104); // 13 * 8 = 104 bits

        // 100 bytes (96 + 4 remainder)
        let a: [u8; 100] = [0; 100];
        let b: [u8; 100] = [0xFF; 100];
        assert_eq!(hamming(&a, &b), 800); // 100 * 8 = 800 bits

        // Also test batch with odd size
        let source: [u8; 13] = [0; 13];
        let targets = vec![[0xFFu8; 13], [0u8; 13]];
        let mut out = vec![0u32; 2];
        hamming_batch(&source, &targets, &mut out);
        assert_eq!(out[0], 104);
        assert_eq!(out[1], 0);
    }
}

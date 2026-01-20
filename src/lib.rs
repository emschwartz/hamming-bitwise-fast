//! A fast, zero-dependency implementation of bitwise Hamming Distance using
//! a method amenable to auto-vectorization.
//!
//! # Quick Start
//!
//! For byte slices (variable-length):
//! ```
//! use hamming_bitwise_fast::hamming_bitwise_slice;
//!
//! let a = [0u8; 128];
//! let b = [0xFFu8; 128];
//! let distance = hamming_bitwise_slice(&a, &b);
//! ```
//!
//! For fixed-size embeddings (10-100% faster for sizes under 2048 bits):
//! ```
//! use hamming_bitwise_fast::hamming_bitwise_array;
//!
//! // 1024-bit embeddings = 128 bytes
//! let a: [u8; 128] = [0; 128];
//! let b: [u8; 128] = [0xFF; 128];
//! let distance = hamming_bitwise_array(&a, &b);
//! ```
//!
//! For batch operations:
//! ```
//! use hamming_bitwise_fast::hamming_bitwise_batch;
//!
//! let source: [u8; 128] = [0; 128];
//! let targets: Vec<[u8; 128]> = vec![[1; 128], [2; 128], [3; 128]];
//! let mut distances = vec![0u32; targets.len()];
//! hamming_bitwise_batch(&source, &targets, &mut distances);
//! ```
//!
//! # Feature Flags
//!
//! - `multiversion_x86`: Enables runtime CPU dispatch for optimal SIMD on x86.
//!   Recommended for x86 deployments. On ARM and other non-x86 platforms,
//!   this feature has no effect (auto-vectorization is already near-optimal).

// ============================================================================
// Public API
// ============================================================================

/// Calculate the bitwise Hamming distance between two byte slices.
///
/// This function uses runtime CPU detection to dispatch to optimized SIMD implementations
/// on x86/x86_64 platforms when the `multiversion_x86` feature is enabled.
///
/// # Performance
///
/// - On x86 with `multiversion_x86`: Uses AVX-512 VPOPCNTDQ when available (~3-5ns for 1024-bit)
/// - On ARM: Uses NEON-friendly auto-vectorization (~3ns for 1024-bit)
/// - On x86 without `multiversion_x86`: Uses auto-vectorized u64 chunked processing (~10ns for 1024-bit)
///
/// For known compile-time sizes, consider [`hamming_bitwise_array`] which is faster
/// for embeddings under 2048 bits (256 bytes):
/// - **ARM**: 10-100% faster (greatest benefit at smaller sizes)
/// - **x86**: 10-30% faster for sizes under 2048 bits
/// - **2048+ bits**: Performance is essentially identical
///
/// # Panics
///
/// Panics if the two slices are not the same length.
#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[multiversion::multiversion(targets(
    "x86_64+avx512vpopcntdq+avx512vl",
    "x86_64+avx512bw+avx512vl",
    "x86_64+avx2+popcnt",
    "x86_64+sse4.2+popcnt",
    "x86+avx2+popcnt",
    "x86+sse4.2+popcnt",
))]
#[inline]
pub fn hamming_bitwise_slice(a: &[u8], b: &[u8]) -> u32 {
    hamming_slice_chunks(a, b)
}

/// Calculate the bitwise Hamming distance between two byte slices.
#[cfg(all(
    not(feature = "multiversion_x86"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[inline]
pub fn hamming_bitwise_slice(a: &[u8], b: &[u8]) -> u32 {
    hamming_slice_chunks(a, b)
}

/// Calculate the bitwise Hamming distance between two byte slices.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
pub fn hamming_bitwise_slice(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Compute Hamming distance for fixed-size byte arrays.
///
/// This is the recommended function for embeddings of known size at compile time,
/// especially for sizes under 2048 bits (256 bytes) where it provides measurable
/// performance benefits over [`hamming_bitwise_slice`].
///
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
/// Compile-time size knowledge enables better loop unrolling and eliminates bounds checking.
///
/// **vs slice (when to prefer arrays):**
/// - **Under 2048 bits**: 10-100% faster depending on platform and size
/// - **2048+ bits**: Similar performance; use whichever is more convenient
///
/// **Absolute timings (1024-bit / 128 bytes):**
/// - ARM (Apple M2, Graviton): ~2.4ns
/// - x86 with `multiversion_x86` (AVX-512): ~3.3ns
/// - x86 without features: ~8.9ns
///
/// # Example
///
/// ```
/// use hamming_bitwise_fast::hamming_bitwise_array;
///
/// // 1024-bit embeddings = 128 bytes
/// let a: [u8; 128] = [0x12; 128];
/// let b: [u8; 128] = [0xFE; 128];
/// let distance = hamming_bitwise_array(&a, &b);
/// ```
#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[multiversion::multiversion(targets(
    "x86_64+avx512vpopcntdq+avx512vl",
    "x86_64+avx512bw+avx512vl",
    "x86_64+avx2+popcnt",
    "x86_64+sse4.2+popcnt",
    "x86+avx2+popcnt",
    "x86+sse4.2+popcnt",
))]
#[inline]
pub fn hamming_bitwise_array<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    hamming_array_chunks(a, b)
}

/// Compute Hamming distance for fixed-size byte arrays (x86 without multiversion).
#[cfg(all(
    not(feature = "multiversion_x86"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[inline]
pub fn hamming_bitwise_array<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    hamming_array_chunks(a, b)
}

/// Compute Hamming distance for fixed-size byte arrays (non-x86).
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
pub fn hamming_bitwise_array<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Compute Hamming distance from one source embedding to many targets.
///
/// This is significantly faster than calling [`hamming_bitwise_array`] in a loop because:
/// 1. The function call overhead is amortized across all comparisons
/// 2. The source embedding can stay in registers
/// 3. With `multiversion_x86`, the CPU dispatch happens once for all comparisons
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
/// use hamming_bitwise_fast::hamming_bitwise_batch;
///
/// let source: [u8; 128] = [0; 128];
/// let targets = vec![[1u8; 128], [2u8; 128], [3u8; 128]];
/// let mut distances = vec![0u32; 3];
///
/// hamming_bitwise_batch(&source, &targets, &mut distances);
/// ```
#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[multiversion::multiversion(targets(
    "x86_64+avx512vpopcntdq+avx512vl",
    "x86_64+avx512bw+avx512vl",
    "x86_64+avx2+popcnt",
    "x86_64+sse4.2+popcnt",
    "x86+avx2+popcnt",
    "x86+sse4.2+popcnt",
))]
pub fn hamming_bitwise_batch<const N: usize>(
    source: &[u8; N],
    targets: &[[u8; N]],
    out: &mut [u32],
) {
    hamming_batch_chunks(source, targets, out)
}

/// Compute Hamming distance from one source to many targets (non-multiversion).
#[cfg(not(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
)))]
pub fn hamming_bitwise_batch<const N: usize>(
    source: &[u8; N],
    targets: &[[u8; N]],
    out: &mut [u32],
) {
    assert_eq!(targets.len(), out.len());

    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        *dist = hamming_bitwise_array(source, target);
    }
}

/// Compute Hamming distance from one source slice to many target slices.
///
/// This is the slice-based equivalent of [`hamming_bitwise_batch`], useful when
/// embedding sizes are not known at compile time or when working with dynamically
/// sized data.
///
/// # Arguments
///
/// * `source` - The source embedding to compare against all targets
/// * `targets` - Slice of target embeddings (each target must have same length as source)
/// * `out` - Output buffer for distances (must have same length as `targets`)
///
/// # Panics
///
/// Panics if:
/// - `out.len() != targets.len()`
/// - Any target has a different length than `source`
///
/// # Example
///
/// ```
/// use hamming_bitwise_fast::hamming_bitwise_batch_slice;
///
/// let source = vec![0u8; 128];
/// let targets_owned: Vec<Vec<u8>> = vec![vec![1u8; 128], vec![2u8; 128], vec![3u8; 128]];
/// let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();
/// let mut distances = vec![0u32; 3];
///
/// hamming_bitwise_batch_slice(&source, &targets, &mut distances);
/// ```
#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[multiversion::multiversion(targets(
    "x86_64+avx512vpopcntdq+avx512vl",
    "x86_64+avx512bw+avx512vl",
    "x86_64+avx2+popcnt",
    "x86_64+sse4.2+popcnt",
    "x86+avx2+popcnt",
    "x86+sse4.2+popcnt",
))]
pub fn hamming_bitwise_batch_slice(source: &[u8], targets: &[&[u8]], out: &mut [u32]) {
    hamming_batch_slice_chunks(source, targets, out)
}

/// Compute Hamming distance from one source slice to many target slices (x86 without multiversion).
#[cfg(all(
    not(feature = "multiversion_x86"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub fn hamming_bitwise_batch_slice(source: &[u8], targets: &[&[u8]], out: &mut [u32]) {
    hamming_batch_slice_chunks(source, targets, out)
}

/// Compute Hamming distance from one source slice to many target slices (non-x86).
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub fn hamming_bitwise_batch_slice(source: &[u8], targets: &[&[u8]], out: &mut [u32]) {
    assert_eq!(targets.len(), out.len());

    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        *dist = hamming_bitwise_slice(source, target);
    }
}

/// Deprecated: Use [`hamming_bitwise_slice`] instead, or consider
/// [`hamming_bitwise_array`] for fixed-size arrays under 2048 bits (10-100% faster) or
/// [`hamming_bitwise_batch`] for comparing one source against many targets.
#[deprecated(
    since = "1.1.0",
    note = "renamed to hamming_bitwise_slice; consider hamming_bitwise_array for fixed-size arrays under 2048 bits (10-100% faster) or hamming_bitwise_batch for bulk comparisons"
)]
#[inline]
pub fn hamming_bitwise_fast(x: &[u8], y: &[u8]) -> u32 {
    hamming_bitwise_slice(x, y)
}

// ============================================================================
// Internal implementations
// ============================================================================

/// Slice implementation using u64 chunks (x86 only).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn hamming_slice_chunks(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());

    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);

    // Process 8 bytes at a time using u64
    let chunks_distance: u32 = a_chunks
        .clone()
        .zip(b_chunks.clone())
        .map(|(a_chunk, b_chunk)| {
            let a_val = u64::from_ne_bytes(a_chunk.try_into().unwrap());
            let b_val = u64::from_ne_bytes(b_chunk.try_into().unwrap());
            (a_val ^ b_val).count_ones()
        })
        .sum();

    // Handle remainder bytes by packing into a u64 for a single popcount
    let remainder_distance = if a.len() % 8 != 0 {
        let a_rem = a_chunks.remainder();
        let b_rem = b_chunks.remainder();
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

/// Array implementation using u64 chunks (x86 only).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn hamming_array_chunks<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);

    let chunks_distance: u32 = a_chunks
        .zip(b_chunks)
        .map(|(a_chunk, b_chunk)| {
            let a_val = u64::from_ne_bytes(a_chunk.try_into().unwrap());
            let b_val = u64::from_ne_bytes(b_chunk.try_into().unwrap());
            (a_val ^ b_val).count_ones()
        })
        .sum();

    // Handle remainder bytes by packing into a u64 for a single popcount.
    // Compiler optimizes this away when N % 8 == 0.
    let remainder_distance = if N % 8 != 0 {
        let a_rem = a.chunks_exact(8).remainder();
        let b_rem = b.chunks_exact(8).remainder();
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

/// Batch implementation using u64 chunks (x86 only).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn hamming_batch_chunks<const N: usize>(source: &[u8; N], targets: &[[u8; N]], out: &mut [u32]) {
    assert_eq!(targets.len(), out.len());

    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        *dist = hamming_array_chunks(source, target);
    }
}

/// Batch slice implementation using u64 chunks (x86 only).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn hamming_batch_slice_chunks(source: &[u8], targets: &[&[u8]], out: &mut [u32]) {
    assert_eq!(targets.len(), out.len());

    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        *dist = hamming_slice_chunks(source, target);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamming_bitwise_slice_correctness() {
        let a = [0u8; 128];
        let b = [0xFFu8; 128];
        assert_eq!(hamming_bitwise_slice(&a, &b), 1024);
        assert_eq!(hamming_bitwise_slice(&a, &a), 0);

        let mut c = [0u8; 128];
        c[0] = 1;
        assert_eq!(hamming_bitwise_slice(&a, &c), 1);
    }

    #[test]
    fn hamming_bitwise_array_correctness() {
        let a: [u8; 128] = [0; 128];
        let b: [u8; 128] = [0xFF; 128];
        assert_eq!(hamming_bitwise_array(&a, &b), 1024);
        assert_eq!(hamming_bitwise_array(&a, &a), 0);

        let mut c = [0u8; 128];
        c[0] = 1;
        assert_eq!(hamming_bitwise_array(&a, &c), 1);
    }

    #[test]
    fn hamming_bitwise_batch_correctness() {
        let source: [u8; 128] = [0; 128];
        let targets = vec![
            [0xFFu8; 128], // 1024 bits different
            [0u8; 128],    // 0 bits different
            [1u8; 128],    // 128 bits different (one bit per byte)
        ];
        let mut out = vec![0u32; 3];

        hamming_bitwise_batch(&source, &targets, &mut out);

        assert_eq!(out[0], 1024);
        assert_eq!(out[1], 0);
        assert_eq!(out[2], 128);
    }

    #[test]
    fn hamming_bitwise_array_matches_slice() {
        let a: [u8; 128] = std::array::from_fn(|i| i as u8);
        let b: [u8; 128] = std::array::from_fn(|i| (i + 128) as u8);

        assert_eq!(hamming_bitwise_array(&a, &b), hamming_bitwise_slice(&a, &b));
    }

    #[test]
    fn different_embedding_sizes() {
        // 512-bit (64 bytes)
        let a: [u8; 64] = [0; 64];
        let b: [u8; 64] = [0xFF; 64];
        assert_eq!(hamming_bitwise_array(&a, &b), 512);

        // 768-bit (96 bytes)
        let a: [u8; 96] = [0; 96];
        let b: [u8; 96] = [0xFF; 96];
        assert_eq!(hamming_bitwise_array(&a, &b), 768);

        // 2048-bit (256 bytes)
        let a: [u8; 256] = [0; 256];
        let b: [u8; 256] = [0xFF; 256];
        assert_eq!(hamming_bitwise_array(&a, &b), 2048);
    }

    #[test]
    fn odd_sizes_with_remainder() {
        // 7 bytes (not a multiple of 8) - tests remainder handling
        let a: [u8; 7] = [0; 7];
        let b: [u8; 7] = [0xFF; 7];
        assert_eq!(hamming_bitwise_array(&a, &b), 56); // 7 * 8 = 56 bits

        // 13 bytes (8 + 5 remainder)
        let a: [u8; 13] = [0; 13];
        let b: [u8; 13] = [0xFF; 13];
        assert_eq!(hamming_bitwise_array(&a, &b), 104); // 13 * 8 = 104 bits

        // 100 bytes (96 + 4 remainder)
        let a: [u8; 100] = [0; 100];
        let b: [u8; 100] = [0xFF; 100];
        assert_eq!(hamming_bitwise_array(&a, &b), 800); // 100 * 8 = 800 bits

        // Also test batch with odd size
        let source: [u8; 13] = [0; 13];
        let targets = vec![[0xFFu8; 13], [0u8; 13]];
        let mut out = vec![0u32; 2];
        hamming_bitwise_batch(&source, &targets, &mut out);
        assert_eq!(out[0], 104);
        assert_eq!(out[1], 0);
    }

    #[test]
    fn hamming_bitwise_batch_slice_correctness() {
        let source = vec![0u8; 128];
        let targets_owned = vec![
            vec![0xFFu8; 128], // 1024 bits different
            vec![0u8; 128],    // 0 bits different
            vec![1u8; 128],    // 128 bits different (one bit per byte)
        ];
        let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();
        let mut out = vec![0u32; 3];

        hamming_bitwise_batch_slice(&source, &targets, &mut out);

        assert_eq!(out[0], 1024);
        assert_eq!(out[1], 0);
        assert_eq!(out[2], 128);

        // Verify results match individual slice calls
        for (i, target) in targets.iter().enumerate() {
            assert_eq!(out[i], hamming_bitwise_slice(&source, target));
        }
    }

    #[test]
    fn hamming_bitwise_batch_slice_empty() {
        let source = vec![0u8; 128];
        let targets: Vec<&[u8]> = vec![];
        let mut out: Vec<u32> = vec![];

        // Should succeed with empty inputs
        hamming_bitwise_batch_slice(&source, &targets, &mut out);
    }

    #[test]
    #[should_panic]
    fn hamming_bitwise_batch_slice_output_size_mismatch() {
        let source = vec![0u8; 128];
        let targets_owned = vec![vec![0u8; 128], vec![0u8; 128]];
        let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();
        let mut out = vec![0u32; 1]; // Wrong size!

        hamming_bitwise_batch_slice(&source, &targets, &mut out);
    }
}

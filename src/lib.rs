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
//! // 1024-bit embeddings = 16 u64s
//! let a: [u64; 16] = [0; 16];
//! let b: [u64; 16] = [u64::MAX; 16];
//! let distance = hamming(&a, &b);
//! ```
//!
//! For batch operations (70% faster than individual calls):
//! ```
//! use hamming_bitwise_fast::hamming_batch;
//!
//! let source: [u64; 16] = [0; 16];
//! let targets: Vec<[u64; 16]> = vec![[1; 16], [2; 16], [3; 16]];
//! let mut distances = vec![0u32; targets.len()];
//! hamming_batch(&source, &targets, &mut distances);
//! ```
//!
//! # Feature Flags
//!
//! - `multiversion`: Enables runtime CPU dispatch for optimal SIMD on x86.
//!   Recommended for x86 deployments. On ARM, auto-vectorization is already
//!   near-optimal, so this adds minimal benefit.
//!
//! - `no-unsafe`: Disables the ARM-specific zero-cost cast optimization.
//!   Use this if you require pure safe Rust code. Performance impact is ~40%
//!   slower on ARM; no impact on x86.

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

// ============================================================================
// Platform-optimized const-generic implementations
// ============================================================================

/// Zero-cost cast from `&[u64; N]` to `&[u8; N*8]` on ARM.
///
/// On ARM (aarch64), u8-based processing is faster due to NEON's byte-level
/// operations. This cast allows us to accept u64 arrays (convenient for users)
/// while processing them as u8 arrays internally.
#[cfg(all(target_arch = "aarch64", not(feature = "no-unsafe")))]
#[inline]
fn as_bytes<const N: usize>(arr: &[u64; N]) -> &[u8] {
    // SAFETY: [u64; N] and [u8; N*8] have the same memory layout.
    // u8 has alignment 1, so any pointer is valid for u8.
    // The lifetime of the returned slice is tied to the input reference.
    unsafe { std::slice::from_raw_parts(arr.as_ptr() as *const u8, N * 8) }
}

/// Compute Hamming distance for fixed-size u64 arrays.
///
/// This is the recommended function for embeddings of known size at compile time.
/// The const generic `N` represents the number of u64 values (not bytes).
///
/// Common sizes:
/// - `N=8`: 512-bit embedding (64 bytes)
/// - `N=12`: 768-bit embedding (96 bytes)
/// - `N=16`: 1024-bit embedding (128 bytes)
/// - `N=32`: 2048-bit embedding (256 bytes)
///
/// # Performance
///
/// - On ARM (M2, Graviton): Uses optimized u8-based processing (~2.5ns for 1024-bit)
/// - On x86 (Intel/AMD): Uses u64-based processing with POPCNT (~11ns for 1024-bit)
///
/// With the `multiversion` feature enabled, x86 performance improves to ~4ns
/// through runtime CPU dispatch to AVX2/AVX-512 instructions.
///
/// # Example
///
/// ```
/// use hamming_bitwise_fast::hamming;
///
/// // 1024-bit embeddings
/// let a: [u64; 16] = [0x123456789ABCDEF0; 16];
/// let b: [u64; 16] = [0xFEDCBA9876543210; 16];
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
pub fn hamming<const N: usize>(a: &[u64; N], b: &[u64; N]) -> u32 {
    hamming_inner(a, b)
}

/// Compute Hamming distance for fixed-size u64 arrays (non-multiversion).
#[cfg(not(feature = "multiversion"))]
#[inline]
pub fn hamming<const N: usize>(a: &[u64; N], b: &[u64; N]) -> u32 {
    hamming_inner(a, b)
}

/// Internal implementation that selects the optimal strategy per platform.
#[inline]
fn hamming_inner<const N: usize>(a: &[u64; N], b: &[u64; N]) -> u32 {
    // On ARM without no-unsafe: use u8 processing (fastest)
    #[cfg(all(target_arch = "aarch64", not(feature = "no-unsafe")))]
    {
        let a_bytes = as_bytes(a);
        let b_bytes = as_bytes(b);
        a_bytes
            .iter()
            .zip(b_bytes.iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum()
    }

    // On x86 or ARM with no-unsafe: use u64 processing
    #[cfg(any(not(target_arch = "aarch64"), feature = "no-unsafe"))]
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
/// # Performance
///
/// Batch operations are approximately 70% faster than individual calls
/// when processing many embeddings.
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
/// let source: [u64; 16] = [0; 16];
/// let targets = vec![[1u64; 16], [2u64; 16], [3u64; 16]];
/// let mut distances = vec![0u32; 3];
///
/// hamming_batch(&source, &targets, &mut distances);
///
/// // distances[i] contains hamming(&source, &targets[i])
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
pub fn hamming_batch<const N: usize>(source: &[u64; N], targets: &[[u64; N]], out: &mut [u32]) {
    hamming_batch_inner(source, targets, out)
}

/// Compute Hamming distance from one source embedding to many targets (non-multiversion).
#[cfg(not(feature = "multiversion"))]
pub fn hamming_batch<const N: usize>(source: &[u64; N], targets: &[[u64; N]], out: &mut [u32]) {
    hamming_batch_inner(source, targets, out)
}

/// Internal batch implementation.
#[inline]
fn hamming_batch_inner<const N: usize>(
    source: &[u64; N],
    targets: &[[u64; N]],
    out: &mut [u32],
) {
    assert_eq!(targets.len(), out.len());

    // On ARM without no-unsafe: use u8 processing
    #[cfg(all(target_arch = "aarch64", not(feature = "no-unsafe")))]
    {
        let source_bytes = as_bytes(source);
        for (i, target) in targets.iter().enumerate() {
            let target_bytes = as_bytes(target);
            out[i] = source_bytes
                .iter()
                .zip(target_bytes.iter())
                .map(|(x, y)| (x ^ y).count_ones())
                .sum();
        }
    }

    // On x86 or ARM with no-unsafe: use u64 processing
    #[cfg(any(not(target_arch = "aarch64"), feature = "no-unsafe"))]
    {
        for (i, target) in targets.iter().enumerate() {
            let mut dist = 0u32;
            for j in 0..N {
                dist += (source[j] ^ target[j]).count_ones();
            }
            out[i] = dist;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamming_bitwise_fast_correctness() {
        // Test with known values
        let a = [0u8; 128];
        let b = [0xFFu8; 128];
        // Every bit differs: 128 bytes * 8 bits = 1024 bits
        assert_eq!(hamming_bitwise_fast(&a, &b), 1024);

        // Same values = 0 distance
        assert_eq!(hamming_bitwise_fast(&a, &a), 0);

        // Single bit difference
        let mut c = [0u8; 128];
        c[0] = 1; // One bit set
        assert_eq!(hamming_bitwise_fast(&a, &c), 1);
    }

    #[test]
    fn hamming_const_generic_correctness() {
        let a: [u64; 16] = [0; 16];
        let b: [u64; 16] = [u64::MAX; 16];
        // Every bit differs: 16 * 64 = 1024 bits
        assert_eq!(hamming(&a, &b), 1024);

        // Same values = 0 distance
        assert_eq!(hamming(&a, &a), 0);

        // Single bit difference
        let mut c = [0u64; 16];
        c[0] = 1;
        assert_eq!(hamming(&a, &c), 1);
    }

    #[test]
    fn hamming_batch_correctness() {
        let source: [u64; 16] = [0; 16];
        let targets = vec![
            [u64::MAX; 16], // 1024 bits different
            [0u64; 16],     // 0 bits different
            [1u64; 16],     // 16 bits different (one bit per u64)
        ];
        let mut out = vec![0u32; 3];

        hamming_batch(&source, &targets, &mut out);

        assert_eq!(out[0], 1024);
        assert_eq!(out[1], 0);
        assert_eq!(out[2], 16);
    }

    #[test]
    fn hamming_matches_bitwise_fast() {
        // Verify const-generic version matches slice version
        let a_bytes: Vec<u8> = (0..128).collect();
        let b_bytes: Vec<u8> = (128..256).map(|x| x as u8).collect();

        let expected = hamming_bitwise_fast(&a_bytes, &b_bytes);

        // Convert to u64 arrays
        let a_u64: [u64; 16] = std::array::from_fn(|i| {
            u64::from_ne_bytes(a_bytes[i * 8..(i + 1) * 8].try_into().unwrap())
        });
        let b_u64: [u64; 16] = std::array::from_fn(|i| {
            u64::from_ne_bytes(b_bytes[i * 8..(i + 1) * 8].try_into().unwrap())
        });

        assert_eq!(hamming(&a_u64, &b_u64), expected);
    }

    #[test]
    fn different_embedding_sizes() {
        // Test 512-bit (8 u64s)
        let a: [u64; 8] = [0; 8];
        let b: [u64; 8] = [u64::MAX; 8];
        assert_eq!(hamming(&a, &b), 512);

        // Test 768-bit (12 u64s)
        let a: [u64; 12] = [0; 12];
        let b: [u64; 12] = [u64::MAX; 12];
        assert_eq!(hamming(&a, &b), 768);

        // Test 2048-bit (32 u64s)
        let a: [u64; 32] = [0; 32];
        let b: [u64; 32] = [u64::MAX; 32];
        assert_eq!(hamming(&a, &b), 2048);
    }
}

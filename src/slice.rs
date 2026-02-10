//! Variable-length slice APIs for bitwise Hamming distance.
//!
//! Use this module when vector sizes are determined at runtime. When the size
//! is known at compile time, prefer the [`array`](crate::array) module for
//! better performance (see [Choosing an API](crate#choosing-an-api)).

// ============================================================================
// Public API
// ============================================================================

/// Compute the bitwise Hamming distance between two byte slices.
///
/// # Panics
///
/// Panics if the slices have different lengths.
///
/// # Example
///
/// ```
/// use hamming_bitwise_fast::slice;
///
/// let a = vec![0xFFu8; 128];
/// let b = vec![0x00u8; 128];
/// let distance = slice::distance(&a, &b);  // 1024
/// ```
#[cfg_attr(
    all(
        feature = "multiversion_x86",
        any(target_arch = "x86", target_arch = "x86_64")
    ),
    multiversion::multiversion(targets(
        "x86_64+avx512vpopcntdq+avx512vl+popcnt",
        "x86_64+avx512bw+avx512vl+popcnt",
        "x86_64+avx2+popcnt",
        "x86_64+sse4.2+popcnt",
        "x86+avx2+popcnt",
        "x86+sse4.2+popcnt",
    ))
)]
#[inline(always)]
pub fn distance(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    crate::distance_impl(a, b)
}

/// Compute Hamming distance from one source slice to many target slices
/// (one-to-many).
///
/// Faster than calling [`distance`] in a loop for one-to-many comparisons.
///
/// # Panics
///
/// Panics if `out.len() != targets.len()` or any target has a different
/// length than `source`.
///
/// # Example
///
/// ```
/// use hamming_bitwise_fast::slice;
///
/// let source = vec![0u8; 128];
/// let targets_owned: Vec<Vec<u8>> = vec![vec![1u8; 128], vec![2u8; 128]];
/// let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();
/// let mut distances = vec![0u32; 2];  // pre-allocate and reuse
///
/// slice::batch(&source, &targets, &mut distances);
/// ```
#[cfg_attr(
    all(
        feature = "multiversion_x86",
        any(target_arch = "x86", target_arch = "x86_64")
    ),
    multiversion::multiversion(targets(
        "x86_64+avx512vpopcntdq+avx512vl+popcnt",
        "x86_64+avx512bw+avx512vl+popcnt",
        "x86_64+avx2+popcnt",
        "x86_64+sse4.2+popcnt",
        "x86+avx2+popcnt",
        "x86+sse4.2+popcnt",
    ))
)]
#[inline(always)]
pub fn batch(source: &[u8], targets: &[&[u8]], out: &mut [u32]) {
    assert_eq!(targets.len(), out.len());

    // For slices, the data layout (&[&[u8]]) isn't contiguous, so gather
    // instructions aren't a concern.
    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        assert_eq!(source.len(), target.len());
        *dist = crate::distance_impl(source, target);
    }
}

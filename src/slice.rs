//! Variable-length slice APIs for bitwise Hamming distance.
//!
//! Use this module when vector sizes are determined at runtime. When the size
//! is known at compile time, prefer the [`array`](crate::array) module for
//! better performance.

/// Block size for early-exit threshold checks (in bytes).
/// See `array` module and `benches/threshold_block_size.rs` for rationale.
const THRESHOLD_BLOCK_SIZE: usize = 32;

// ============================================================================
// Private implementation functions
// ============================================================================

/// x86 slice implementation using u64 chunks for auto-vectorization.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn slice_impl(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);

    let main: u32 = a_chunks
        .clone()
        .zip(b_chunks.clone())
        .map(|(a, b)| {
            let a = u64::from_ne_bytes(a.try_into().unwrap());
            let b = u64::from_ne_bytes(b.try_into().unwrap());
            (a ^ b).count_ones()
        })
        .sum();

    let rem: u32 = a_chunks
        .remainder()
        .iter()
        .zip(b_chunks.remainder())
        .map(|(a, b)| (a ^ b).count_ones())
        .sum();

    main + rem
}

/// Non-x86 slice implementation using simple byte iteration.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline(always)]
fn slice_impl(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// x86 slice threshold: u64 chunking within each block for VPOPCNTDQ/POPCNT.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn slice_threshold_impl(a: &[u8], b: &[u8], threshold: u32) -> Option<u32> {
    assert_eq!(a.len(), b.len());
    let mut distance: u32 = 0;

    let a_blocks = a.chunks_exact(THRESHOLD_BLOCK_SIZE);
    let b_blocks = b.chunks_exact(THRESHOLD_BLOCK_SIZE);
    let a_rem = a_blocks.remainder();
    let b_rem = b_blocks.remainder();

    for (a_block, b_block) in a_blocks.zip(b_blocks) {
        let a_arr: &[u8; THRESHOLD_BLOCK_SIZE] = a_block.try_into().unwrap();
        let b_arr: &[u8; THRESHOLD_BLOCK_SIZE] = b_block.try_into().unwrap();

        let block_dist: u32 = a_arr
            .chunks_exact(8)
            .zip(b_arr.chunks_exact(8))
            .map(|(a_chunk, b_chunk)| {
                let a_val = u64::from_ne_bytes(a_chunk.try_into().unwrap());
                let b_val = u64::from_ne_bytes(b_chunk.try_into().unwrap());
                (a_val ^ b_val).count_ones()
            })
            .sum();

        distance += block_dist;
        if distance > threshold {
            return None;
        }
    }

    let rem_dist: u32 = a_rem
        .iter()
        .zip(b_rem.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum();
    distance += rem_dist;

    if distance <= threshold {
        Some(distance)
    } else {
        None
    }
}

/// Non-x86 slice threshold: byte iteration for NEON cnt.16b auto-vectorization.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline(always)]
fn slice_threshold_impl(a: &[u8], b: &[u8], threshold: u32) -> Option<u32> {
    assert_eq!(a.len(), b.len());
    let mut distance: u32 = 0;

    let a_blocks = a.chunks_exact(THRESHOLD_BLOCK_SIZE);
    let b_blocks = b.chunks_exact(THRESHOLD_BLOCK_SIZE);
    let a_rem = a_blocks.remainder();
    let b_rem = b_blocks.remainder();

    for (a_block, b_block) in a_blocks.zip(b_blocks) {
        let a_arr: &[u8; THRESHOLD_BLOCK_SIZE] = a_block.try_into().unwrap();
        let b_arr: &[u8; THRESHOLD_BLOCK_SIZE] = b_block.try_into().unwrap();

        let block_dist: u32 = a_arr
            .iter()
            .zip(b_arr.iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum();

        distance += block_dist;
        if distance > threshold {
            return None;
        }
    }

    let rem_dist: u32 = a_rem
        .iter()
        .zip(b_rem.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum();
    distance += rem_dist;

    if distance <= threshold {
        Some(distance)
    } else {
        None
    }
}

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
    all(feature = "multiversion_x86", any(target_arch = "x86", target_arch = "x86_64")),
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
    slice_impl(a, b)
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
    all(feature = "multiversion_x86", any(target_arch = "x86", target_arch = "x86_64")),
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
    // instructions aren't a concern. Calling slice_impl directly is faster.
    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        *dist = slice_impl(source, target);
    }
}

/// Compute Hamming distance with early exit when distance exceeds a threshold.
///
/// Returns `Some(distance)` if the distance is `<= max`, or `None` if it
/// exceeds the threshold. Internally checks the running distance every 256
/// bits (32 bytes).
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
/// let a = vec![0u8; 128];
/// let b = vec![0xFFu8; 128];  // distance = 1024
///
/// assert_eq!(slice::threshold(&a, &b, 100), None);
/// assert_eq!(slice::threshold(&a, &b, 2000), Some(1024));
/// ```
#[cfg_attr(
    all(feature = "multiversion_x86", any(target_arch = "x86", target_arch = "x86_64")),
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
pub fn threshold(a: &[u8], b: &[u8], max: u32) -> Option<u32> {
    slice_threshold_impl(a, b, max)
}

/// Batch Hamming distance with early exit: compare one source against many
/// targets, skipping comparisons that exceed `max`.
///
/// Targets that exceed the threshold are written as [`u32::MAX`] in the
/// output. Returns the minimum distance found (or [`u32::MAX`] if all
/// targets exceeded the threshold).
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
/// let targets_owned = vec![vec![0xFFu8; 128], vec![0u8; 128], vec![1u8; 128]];
/// let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();
/// let mut distances = vec![0u32; 3];
///
/// let best = slice::batch_threshold(
///     &source, &targets, 500, &mut distances,
/// );
///
/// assert_eq!(distances[0], u32::MAX); // 1024 > 500, rejected
/// assert_eq!(distances[1], 0);        // within threshold
/// assert_eq!(distances[2], 128);      // within threshold
/// assert_eq!(best, 0);                // minimum distance found
/// ```
#[cfg_attr(
    all(feature = "multiversion_x86", any(target_arch = "x86", target_arch = "x86_64")),
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
pub fn batch_threshold(source: &[u8], targets: &[&[u8]], max: u32, out: &mut [u32]) -> u32 {
    assert_eq!(targets.len(), out.len());
    let mut best = u32::MAX;
    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        match slice_threshold_impl(source, target, max) {
            Some(d) => {
                *dist = d;
                if d < best { best = d; }
            }
            None => {
                *dist = u32::MAX;
            }
        }
    }
    best
}

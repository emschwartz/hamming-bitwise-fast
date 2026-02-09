//! Fixed-size array APIs for bitwise Hamming distance.
//!
//! Use this module when the vector size is known at compile time (e.g., 1024-bit
//! embeddings stored as `[u8; 128]`). The compiler can fully unroll and vectorize
//! the loop, yielding the fastest code.

/// Block size for early-exit threshold checks (in bytes).
/// Each block is converted to a fixed-size array for auto-vectorization,
/// then the running distance is checked against the threshold between blocks.
///
/// 32 bytes was chosen via benchmarking (see `benches/threshold_block_size.rs`):
/// it's ~30% faster than 64B on tight thresholds for 1024-bit vectors (the most
/// common embedding size) while performing within ~10% on loose thresholds.
const THRESHOLD_BLOCK_SIZE: usize = 32;

// ============================================================================
// PERFORMANCE INVARIANT: AVX-512 Gather Avoidance
// ============================================================================
//
// batch() and batch_threshold() iterate over &[[u8; N]] — contiguous memory.
// On x86 AVX-512, LLVM can analyze stride patterns across iterations and emit
// VPGATHERQQ gather instructions. These are 2-10x SLOWER than contiguous
// VMOVDQU64 loads because each element requires a separate memory fetch.
//
// Mitigation: wrap the target reference in std::hint::black_box() to make the
// pointer opaque. This prevents stride analysis while preserving all other
// optimizations (loop unrolling, auto-vectorization within each iteration).
//
// This invariant MUST be maintained when modifying batch functions.
// Verify: inspect x86 AVX-512 assembly for absence of VPGATHERQQ.
// Proof: benches/batch_input_type.rs gather_demo (A/B comparison).
// ============================================================================

// ============================================================================
// Private implementation functions
// ============================================================================

/// x86 array implementation using u64 chunks for auto-vectorization.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn array_impl<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
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

/// Non-x86 array implementation using simple byte iteration.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline(always)]
fn array_impl<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// x86 array threshold: u64 chunking within each block for VPOPCNTDQ/POPCNT.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn array_threshold_impl<const N: usize>(
    a: &[u8; N],
    b: &[u8; N],
    threshold: u32,
) -> Option<u32> {
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

    // Remainder (< THRESHOLD_BLOCK_SIZE bytes) — not worth early-exiting from
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

/// Non-x86 array threshold: byte iteration for NEON cnt.16b auto-vectorization.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline(always)]
fn array_threshold_impl<const N: usize>(
    a: &[u8; N],
    b: &[u8; N],
    threshold: u32,
) -> Option<u32> {
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

    // Remainder (< THRESHOLD_BLOCK_SIZE bytes) — not worth early-exiting from
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

/// Compute the bitwise Hamming distance between two fixed-size byte arrays.
///
/// This is the recommended API when the vector size is known at compile time.
/// The compiler can fully unroll the inner loop and emit optimal SIMD
/// instructions for the target platform.
///
/// # Example
///
/// ```
/// use hamming_bitwise_fast::array;
///
/// let a: [u8; 128] = [0x12; 128];  // 1024-bit
/// let b: [u8; 128] = [0xFE; 128];
/// let distance = array::distance(&a, &b);
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
#[inline]
pub fn distance<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    array_impl(a, b)
}

/// Compute Hamming distance from one source to many targets (one-to-many).
///
/// Faster than calling [`distance`] in a loop for one-to-many comparisons.
///
/// # Panics
///
/// Panics if `out.len() != targets.len()`.
///
/// # Example
///
/// ```
/// use hamming_bitwise_fast::array;
///
/// let source: [u8; 128] = [0; 128];
/// let targets = vec![[1u8; 128], [2u8; 128], [3u8; 128]];
/// let mut distances = vec![0u32; 3];  // pre-allocate and reuse
///
/// array::batch(&source, &targets, &mut distances);
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
#[inline]
pub fn batch<const N: usize>(
    source: &[u8; N],
    targets: &[[u8; N]],
    out: &mut [u32],
) {
    assert_eq!(targets.len(), out.len());

    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        // PERFORMANCE INVARIANT: gather avoidance on x86 AVX-512
        //
        // black_box prevents LLVM from analyzing pointer stride patterns across
        // loop iterations of the contiguous &[[u8; N]] layout. Without this,
        // LLVM emits VPGATHERQQ instructions that are 2-10x slower than the
        // contiguous VMOVDQU64 loads we want.
        //
        // This is a first-class Rust optimization barrier (inline assembly) that
        // works under LTO and is independent of multiversion's implementation.
        // See benches/batch_input_type.rs gather_demo for A/B proof benchmarks.
        let target = std::hint::black_box(target);
        *dist = array_impl(source, target);
    }
}

/// Compute Hamming distance with early exit when distance exceeds a threshold.
///
/// Returns `Some(distance)` if the distance is `<= max`, or `None` if it
/// exceeds the threshold. Internally checks the running distance every 256
/// bits (32 bytes).
///
/// **When to use:** Nearest-neighbor search where most candidates are far
/// from the query vector (e.g., threshold is ~10% of max possible distance).
///
/// **When NOT to use:** If most comparisons will pass the threshold, use
/// [`distance`] instead — the early-exit checks add overhead with no benefit.
///
/// # Example
///
/// ```
/// use hamming_bitwise_fast::array;
///
/// let a: [u8; 128] = [0; 128];
/// let b: [u8; 128] = [0xFF; 128];  // distance = 1024
///
/// // Tight threshold: exits early without processing full vector
/// assert_eq!(array::threshold(&a, &b, 100), None);
///
/// // Loose threshold: computes full distance
/// assert_eq!(array::threshold(&a, &b, 2000), Some(1024));
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
#[inline]
pub fn threshold<const N: usize>(
    a: &[u8; N],
    b: &[u8; N],
    max: u32,
) -> Option<u32> {
    array_threshold_impl(a, b, max)
}

/// Batch Hamming distance with early exit: compare one source against many
/// targets, skipping comparisons that exceed `max`.
///
/// Targets that exceed the threshold are written as [`u32::MAX`] in the
/// output. Returns the minimum distance found (or [`u32::MAX`] if all
/// targets exceeded the threshold).
///
/// Designed for streaming top-k search: use the heap's worst entry as the
/// threshold — it tightens as better items are found, causing more early
/// exits over time.
///
/// ```
/// use hamming_bitwise_fast::array;
///
/// // Maintain a heap of top-k items. Use the heap's worst
/// // score as the threshold — it tightens as better items
/// // are found, causing more early exits over time.
/// // let threshold = heap.peek().map_or(u32::MAX, |worst| worst.distance);
/// // let best = array::batch_threshold(&item, &interests, threshold, &mut out);
/// ```
///
/// # Panics
///
/// Panics if `out.len() != targets.len()`.
///
/// # Example
///
/// ```
/// use hamming_bitwise_fast::array;
///
/// let source: [u8; 128] = [0; 128];
/// let targets = vec![[0xFFu8; 128], [0u8; 128], [1u8; 128]];
/// let mut distances = vec![0u32; 3];
///
/// let best = array::batch_threshold(
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
#[inline]
pub fn batch_threshold<const N: usize>(
    source: &[u8; N],
    targets: &[[u8; N]],
    max: u32,
    out: &mut [u32],
) -> u32 {
    assert_eq!(targets.len(), out.len());
    let mut best = u32::MAX;
    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        // PERFORMANCE INVARIANT: gather avoidance (see batch() comment above)
        let target = std::hint::black_box(target);
        match array_threshold_impl(source, target, max) {
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

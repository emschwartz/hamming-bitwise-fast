//! A fast, zero-dependency\* implementation of bitwise Hamming distance using
//! a method amenable to auto-vectorization.
//!
//! _\* Zero dependencies by default. The optional `multiversion_x86` feature adds the
//! [`multiversion`](https://crates.io/crates/multiversion) crate for runtime CPU detection on x86._
//!
//! # Quick Start
//!
//! ```
//! use hamming_bitwise_fast::array;
//!
//! let a: [u8; 128] = [0xFF; 128];  // 1024-bit vectors
//! let b: [u8; 128] = [0x00; 128];
//!
//! // Single comparison
//! let distance = array::distance(&a, &b);  // 1024
//!
//! // One source vs many targets
//! let targets = vec![a, b];
//! let mut distances = vec![0u32; 2];
//! array::batch(&a, &targets, &mut distances);
//! ```
//!
//! # Choosing an API
//!
//! ## Fixed-size arrays vs slices
//!
//! If the vector size is known at compile time (e.g., 1024-bit embeddings are
//! `[u8; 128]`), use the [`mod@array`] module — the compiler can fully unroll and
//! vectorize the loop. Use [`mod@slice`] when sizes vary at runtime.
//!
//! ## Single vs batch
//!
//! Use [`array::batch`] or [`slice::batch`] when comparing one source against
//! many targets. Batch amortizes dispatch overhead and is the fastest approach
//! for one-to-many comparisons.
//!
//! ## Early exit for top-k search (`threshold` / `batch_threshold`)
//!
//! [`array::threshold`] and [`array::batch_threshold`] add an early-exit check: if
//! the running Hamming distance exceeds a threshold partway through the vector,
//! computation stops immediately.
//!
//! **Use `batch_threshold` when:**
//! - Searching for nearest neighbors or top-k closest items
//! - You maintain a threshold (e.g., worst score in a top-k heap)
//! - Most candidates will be rejected (far from the query)
//!
//! The check runs every 512 bits (64 bytes). For 1024-bit vectors, a reject
//! can happen after processing only the first half. With embeddings trained
//! using Matryoshka Representation Learning (MRL), semantic information is
//! concentrated in early bits, making early exit particularly effective.
//!
//! **Use regular `batch` when** you need all distances or most comparisons
//! will pass the threshold.
//!
//! # Platform Behavior
//!
//! | Platform | Configuration | Behavior |
//! |----------|---------------|----------|
//! | x86/x86_64 | `multiversion_x86` feature | Runtime CPU detection (AVX-512/AVX2/SSE4.2) |
//! | x86/x86_64 | Default | Baseline SSE2 only (slow) |
//! | ARM | Default | NEON is baseline; already optimized |
//!
//! On x86, enable `multiversion_x86` for portable binaries that automatically use
//! the best available instructions:
//! ```sh
//! cargo add hamming-bitwise-fast --features multiversion_x86
//! ```
//!
//! Alternatively, compile with `-C target-cpu=native` for fast binaries that only
//! run on CPUs with the same features as the build machine.
//!
//! On ARM (including Apple Silicon), the default build is already fast.
//!
//! # Feature Flags
//!
//! - `multiversion_x86`: Enables runtime CPU detection for optimal SIMD on x86.
//!   Recommended for x86 deployments where you can't use `-C target-cpu=native`.

#[cfg(test)]
mod tests;

pub mod array;
pub mod slice;

// ============================================================================
// Private implementation functions - no multiversion, just #[inline(always)]
// These get inlined into the multiversion-generated functions.
// ============================================================================

/// x86 slice implementation using u64 chunks for auto-vectorization.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn slice_impl(a: &[u8], b: &[u8]) -> u32 {
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

/// x86 array implementation using u64 chunks for auto-vectorization.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn array_impl<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
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
pub(crate) fn slice_impl(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Non-x86 array implementation using simple byte iteration.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline(always)]
pub(crate) fn array_impl<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

// ============================================================================
// Threshold implementation functions
// ============================================================================

/// Block size for early-exit threshold checks (in bytes).
/// Each block is converted to a fixed-size array for auto-vectorization,
/// then the running distance is checked against the threshold between blocks.
const THRESHOLD_BLOCK_SIZE: usize = 64;

/// Block-based early-exit for arrays: process fixed-size blocks with vectorizable
/// inner loops, checking the threshold only between blocks.
///
/// The key trick: each block is converted to `&[u8; THRESHOLD_BLOCK_SIZE]` via
/// `try_into()`, giving the compiler a fixed-size array it can fully vectorize
/// (NEON cnt.16b on ARM, VPOPCNTDQ on x86). The threshold check happens outside
/// the vectorized inner loop, preserving auto-vectorization.
#[inline(always)]
pub(crate) fn array_threshold_impl<const N: usize>(
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
        // Convert to fixed-size array so the compiler can fully unroll + vectorize.
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

/// Block-based early-exit for slices: same algorithm as `array_threshold_impl`
/// but accepts `&[u8]` instead of `&[u8; N]`.
#[inline(always)]
pub(crate) fn slice_threshold_impl(a: &[u8], b: &[u8], threshold: u32) -> Option<u32> {
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
// Public API via macro-generated functions
// ============================================================================

/// Generates platform-specific versions of a function:
/// - x86/x86_64 with `multiversion_x86` feature: runtime CPU dispatch
/// - All other configurations: simple `#[inline(always)]`
macro_rules! define_hamming_fn {
    (
        $(#[$doc:meta])*
        pub fn $name:ident $(<const $N:ident : usize>)? ($($arg:ident : $arg_ty:ty),* $(,)?) $(-> $ret:ty)? $body:block
    ) => {
        $(#[$doc])*
        #[cfg(all(
            feature = "multiversion_x86",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        #[multiversion::multiversion(targets(
            "x86_64+avx512vpopcntdq+avx512vl+popcnt",
            "x86_64+avx512bw+avx512vl+popcnt",
            "x86_64+avx2+popcnt",
            "x86_64+sse4.2+popcnt",
            "x86+avx2+popcnt",
            "x86+sse4.2+popcnt",
        ))]
        #[inline(always)]
        pub fn $name $(<const $N : usize>)? ($($arg : $arg_ty),*) $(-> $ret)? $body

        $(#[$doc])*
        #[cfg(not(all(
            feature = "multiversion_x86",
            any(target_arch = "x86", target_arch = "x86_64")
        )))]
        #[inline(always)]
        pub fn $name $(<const $N : usize>)? ($($arg : $arg_ty),*) $(-> $ret)? $body
    };
}

// Make the macro available to submodules
pub(crate) use define_hamming_fn;

// ============================================================================
// Deprecated compatibility shim
// ============================================================================

/// Deprecated: Use [`slice::distance`] instead, or consider
/// [`array::distance`] for fixed-size arrays or
/// [`array::batch`] for comparing one source against many targets.
#[deprecated(
    since = "1.1.0",
    note = "use hamming_bitwise_fast::slice::distance"
)]
#[inline(always)]
pub fn hamming_bitwise_fast(x: &[u8], y: &[u8]) -> u32 {
    slice::distance(x, y)
}

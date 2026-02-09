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
//! many targets. Batch is the fastest approach for one-to-many comparisons.
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
//! The check runs every 256 bits (32 bytes). For 1024-bit vectors, a reject
//! can happen after processing only the first quarter. With embeddings trained
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
// Shared implementation functions
// ============================================================================

/// Block size for early-exit threshold checks (in bytes).
/// Each block is checked against the threshold between iterations.
///
/// 32 bytes was chosen via benchmarking (see `benches/threshold_block_size.rs`):
/// it's ~30% faster than 64B on tight thresholds for 1024-bit vectors (the most
/// common embedding size) while performing within ~10% on loose thresholds.
pub(crate) const THRESHOLD_BLOCK_SIZE: usize = 32;

/// x86 distance implementation using u64 chunks for auto-vectorization.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn distance_impl(a: &[u8], b: &[u8]) -> u32 {
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

/// Non-x86 distance implementation using simple byte iteration.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline(always)]
pub(crate) fn distance_impl(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Threshold implementation: early-exit when running distance exceeds threshold.
/// Platform-specific logic is delegated to `distance_impl`.
#[inline(always)]
pub(crate) fn threshold_impl(a: &[u8], b: &[u8], threshold: u32) -> Option<u32> {
    let mut distance: u32 = 0;

    let a_blocks = a.chunks_exact(THRESHOLD_BLOCK_SIZE);
    let b_blocks = b.chunks_exact(THRESHOLD_BLOCK_SIZE);
    let a_rem = a_blocks.remainder();
    let b_rem = b_blocks.remainder();

    for (a_block, b_block) in a_blocks.zip(b_blocks) {
        distance += distance_impl(a_block, b_block);
        if distance > threshold {
            return None;
        }
    }

    // Remainder (< THRESHOLD_BLOCK_SIZE bytes) — not worth early-exiting from
    distance += distance_impl(a_rem, b_rem);

    if distance <= threshold {
        Some(distance)
    } else {
        None
    }
}

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

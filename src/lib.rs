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

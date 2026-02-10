//! Fast bitwise Hamming distance using auto-vectorization with runtime SIMD
//! detection on x86.
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
//! `[u8; 128]`), use the [`mod@array`] module for the best performance.
//!
//! Use [`mod@slice`] when sizes vary at runtime or are not known until program
//! execution.
//!
//! ## Single vs batch
//!
//! Use [`array::batch`] or [`slice::batch`] when comparing one source against
//! many targets. Batch is the fastest approach for one-to-many comparisons.
//!
//! # Platform Behavior
//!
//! | Platform | Configuration | Behavior |
//! |----------|---------------|----------|
//! | x86/x86_64 | Default | Runtime CPU detection via [`multiversion`](https://crates.io/crates/multiversion) (AVX-512/AVX2/SSE4.2) |
//! | x86/x86_64 | `default-features = false` | Baseline SSE2 only (slow) |
//! | ARM | Default | NEON is baseline; already optimized |
//!
//! On x86, the default build automatically detects and uses the best available
//! SIMD instructions at runtime:
//! ```sh
//! cargo add hamming-bitwise-fast
//! ```
//!
//! For best single-call performance on x86, enable LTO so the compiler can
//! auto-vectorize across the crate boundary:
//! ```toml
//! [profile.release]
//! lto = true
//! ```
//!
//! For maximum performance, also compile with `-C target-cpu=native`
//! (eliminates runtime dispatch overhead, at the cost of portability).
//!
//! On ARM (including Apple Silicon), the default build is already fast.
//!
//! # Feature Flags
//!
//! - `multiversion_x86` *(enabled by default)*: Enables runtime CPU detection
//!   for optimal SIMD on x86 via the [`multiversion`](https://crates.io/crates/multiversion) crate.
//!   Disable with `default-features = false` if you need zero dependencies or
//!   are targeting a known CPU with `-C target-cpu=native`.

#[cfg(test)]
mod tests;

pub mod array;
pub mod slice;

// ============================================================================
// Shared implementation functions
// ============================================================================

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
            // chunks_exact(8) guarantees exactly 8 bytes per chunk
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

/// Convenience alias for [`slice::distance`] that matches the crate name.
///
/// For fixed-size arrays, consider [`array::distance`] or
/// [`array::batch`] for comparing one source against many targets.
#[inline(always)]
pub fn hamming_bitwise_fast(x: &[u8], y: &[u8]) -> u32 {
    slice::distance(x, y)
}

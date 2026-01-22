//! A fast, zero-dependency\* implementation of bitwise Hamming Distance using
//! a method amenable to auto-vectorization.
//!
//! _\* Zero dependencies by default. The optional `multiversion_x86` feature adds the
//! [`multiversion`](https://crates.io/crates/multiversion) crate for runtime CPU detection on x86._
//!
//! # Quick Start
//!
//! For byte slices (variable-length):
//! ```
//! use hamming_bitwise_fast::{hamming_bitwise_slice, hamming_bitwise_slice_batch};
//!
//! // Single comparison
//! let a = vec![0xFFu8; 128];
//! let b = vec![0x00u8; 128];
//! let distance = hamming_bitwise_slice(&a, &b); // 1024
//!
//! // Batch comparison (one source vs many targets)
//! // Pre-allocate result vec once and reuse across calls for best performance
//! let source = vec![0x00u8; 128];
//! let targets: Vec<&[u8]> = vec![&a, &b];
//! let mut distances = vec![0u32; 2];
//! hamming_bitwise_slice_batch(&source, &targets, &mut distances); // [1024, 0]
//! ```
//!
//! For fixed-size arrays (faster when size is known at compile time):
//! ```
//! use hamming_bitwise_fast::{hamming_bitwise_array, hamming_bitwise_array_batch};
//!
//! // Single comparison
//! let a: [u8; 128] = [0xFF; 128];  // 1024-bit vectors
//! let b: [u8; 128] = [0x00; 128];
//! let distance = hamming_bitwise_array(&a, &b); // 1024
//!
//! // Batch comparison (one source vs many targets)
//! // Pre-allocate result vec once and reuse across calls for best performance
//! let source: [u8; 128] = [0x00; 128];
//! let targets: Vec<[u8; 128]> = vec![a, b];
//! let mut distances = vec![0u32; 2];
//! hamming_bitwise_array_batch(&source, &targets, &mut distances); // [1024, 0]
//! ```
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
//! Batch operations ([`hamming_bitwise_array_batch`], [`hamming_bitwise_slice_batch`])
//! are faster for one-to-many comparisons.
//!
//! # Feature Flags
//!
//! - `multiversion_x86`: Enables runtime CPU detection for optimal SIMD on x86.
//!   Recommended for x86 deployments where you can't use `-C target-cpu=native`.

#[cfg(test)]
mod tests;

// ============================================================================
// Private implementation functions - no multiversion, just #[inline(always)]
// These get inlined into the multiversion-generated functions.
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

/// Non-x86 array implementation using simple byte iteration.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline(always)]
fn array_impl<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
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

define_hamming_fn! {
    /// Compute the bitwise Hamming distance between two byte slices.
    ///
    /// # Panics
    ///
    /// Panics if the slices have different lengths.
    ///
    /// # See Also
    ///
    /// - [`hamming_bitwise_array`] - Faster when size is known at compile time
    /// - [`hamming_bitwise_slice_batch`] - Much faster for one-to-many comparisons
    /// - [Platform Behavior](crate#platform-behavior) - Performance by platform
    pub fn hamming_bitwise_slice(a: &[u8], b: &[u8]) -> u32 {
        slice_impl(a, b)
    }
}

define_hamming_fn! {
    /// Compute Hamming distance for fixed-size byte arrays.
    ///
    /// # See Also
    ///
    /// - [`hamming_bitwise_array_batch`] - Much faster for one-to-many comparisons
    /// - [Platform Behavior](crate#platform-behavior) - Performance by platform
    ///
    /// # Example
    ///
    /// ```
    /// use hamming_bitwise_fast::hamming_bitwise_array;
    ///
    /// let a: [u8; 128] = [0x12; 128];  // 1024-bit
    /// let b: [u8; 128] = [0xFE; 128];
    /// let distance = hamming_bitwise_array(&a, &b);
    /// ```
    pub fn hamming_bitwise_array<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
        array_impl(a, b)
    }
}

define_hamming_fn! {
    /// Compute Hamming distance from one source to many targets (one-to-many).
    ///
    /// On x86 with the `multiversion_x86` feature, this is faster than calling
    /// [`hamming_bitwise_array`] in a loop because CPU dispatch happens once
    /// per batch instead of once per comparison.
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != targets.len()`.
    ///
    /// # See Also
    ///
    /// - [Platform Behavior](crate#platform-behavior) - Performance by platform
    ///
    /// # Example
    ///
    /// ```
    /// use hamming_bitwise_fast::hamming_bitwise_array_batch;
    ///
    /// let source: [u8; 128] = [0; 128];
    /// let targets = vec![[1u8; 128], [2u8; 128], [3u8; 128]];
    /// let mut distances = vec![0u32; 3];  // pre-allocate and reuse
    ///
    /// hamming_bitwise_array_batch(&source, &targets, &mut distances);
    /// ```
    pub fn hamming_bitwise_array_batch<const N: usize>(
        source: &[u8; N],
        targets: &[[u8; N]],
        out: &mut [u32],
    ) {
        assert_eq!(targets.len(), out.len());

        // Call hamming_bitwise_array directly. The multiversion dispatch creates a
        // boundary that prevents the compiler from seeing the contiguous `&[[u8; N]]`
        // layout, avoiding slow VPGATHERQQ gather instructions. This approach is
        // ~16% faster than using black_box to hide the memory layout.
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            *dist = hamming_bitwise_array(source, target);
        }
    }
}

define_hamming_fn! {
    /// Compute Hamming distance from one source slice to many target slices (one-to-many).
    ///
    /// This is the slice-based equivalent of [`hamming_bitwise_array_batch`].
    ///
    /// On x86 with the `multiversion_x86` feature, this is faster than calling
    /// [`hamming_bitwise_slice`] in a loop because CPU dispatch happens once
    /// per batch instead of once per comparison.
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != targets.len()` or any target has a different length than `source`.
    ///
    /// # See Also
    ///
    /// - [`hamming_bitwise_array_batch`] - Faster when size is known at compile time
    /// - [Platform Behavior](crate#platform-behavior) - Performance by platform
    ///
    /// # Example
    ///
    /// ```
    /// use hamming_bitwise_fast::hamming_bitwise_slice_batch;
    ///
    /// let source = vec![0u8; 128];
    /// let targets_owned: Vec<Vec<u8>> = vec![vec![1u8; 128], vec![2u8; 128]];
    /// let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();
    /// let mut distances = vec![0u32; 2];  // pre-allocate and reuse
    ///
    /// hamming_bitwise_slice_batch(&source, &targets, &mut distances);
    /// ```
    pub fn hamming_bitwise_slice_batch(source: &[u8], targets: &[&[u8]], out: &mut [u32]) {
        assert_eq!(targets.len(), out.len());

        // For slices, the data layout (`&[&[u8]]`) isn't contiguous, so the compiler
        // won't use gather instructions. Inlining the impl is faster than calling
        // the MV single function because it avoids dispatch overhead per iteration.
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            *dist = slice_impl(source, target);
        }
    }
}

/// Deprecated: Use [`hamming_bitwise_slice`] instead, or consider
/// [`hamming_bitwise_array`] for fixed-size arrays or
/// [`hamming_bitwise_array_batch`] for comparing one source against many targets.
#[deprecated(
    since = "1.1.0",
    note = "renamed to hamming_bitwise_slice; consider hamming_bitwise_array for fixed-size arrays or hamming_bitwise_array_batch for bulk comparisons"
)]
#[inline(always)]
pub fn hamming_bitwise_fast(x: &[u8], y: &[u8]) -> u32 {
    hamming_bitwise_slice(x, y)
}

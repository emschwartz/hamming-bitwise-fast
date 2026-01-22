//! A fast, zero-dependency\* implementation of bitwise Hamming Distance using
//! a method amenable to auto-vectorization.
//!
//! _\* Zero dependencies by default. The optional `multiversion_x86` feature adds the
//! [`multiversion`](https://crates.io/crates/multiversion) crate for runtime SIMD support detection on x86 CPUs._
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
//! For fixed-size arrays (faster for small sizes):
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
//! # SIMD on x86
//!
//! **TL;DR:** On x86, enable the `multiversion_x86` feature for best performance:
//! ```sh
//! cargo add hamming-bitwise-fast --features multiversion_x86
//! ```
//!
//! Rust targets the baseline x86-64 instruction set (SSE2 only) by default, which is slow
//! for Hamming distance (~10ns for 1024-bit). The `multiversion_x86` feature uses runtime
//! CPU detection to automatically use AVX2/AVX-512 when available (~3-4ns), or even faster
//! with batch operations (~0.8ns per comparison).
//!
//! On ARM, NEON is the baseline so default builds are already fast.
//!
//! For more details, see the [README](https://github.com/emschwartz/hamming-bitwise-fast#simd-on-x86).
//!
//! # Feature Flags
//!
//! - `multiversion_x86`: Enables runtime CPU dispatch for optimal SIMD on x86.
//!   **Strongly recommended** for x86 deployments where you can't use `-C target-cpu=native`.

// ============================================================================
// Public API
// ============================================================================

/// Calculate the bitwise Hamming distance between two byte slices.
///
/// This function uses runtime CPU detection to dispatch to optimized SIMD implementations
/// on x86/x86_64 platforms when the `multiversion_x86` feature is enabled.
///
/// # Performance (1024-bit, Ice Lake x86)
///
/// | Configuration | Speed |
/// |---------------|-------|
/// | Default (no SIMD) | ~10ns |
/// | `multiversion_x86` feature | ~3-4ns |
///
/// On ARM, NEON is the baseline so default builds are already fast.
///
/// For batch comparisons, see [`hamming_bitwise_slice_batch`].
///
/// # Panics
///
/// Panics if the two slices are not the same length.
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
pub fn hamming_bitwise_slice(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);

    let main: u32 = a_chunks.clone().zip(b_chunks.clone())
        .map(|(a, b)| {
            let a = u64::from_ne_bytes(a.try_into().unwrap());
            let b = u64::from_ne_bytes(b.try_into().unwrap());
            (a ^ b).count_ones()
        })
        .sum();

    let rem: u32 = a_chunks.remainder().iter().zip(b_chunks.remainder())
        .map(|(a, b)| (a ^ b).count_ones())
        .sum();

    main + rem
}

/// Calculate the bitwise Hamming distance between two byte slices.
///
/// This function uses runtime CPU detection to dispatch to optimized SIMD implementations
/// on x86/x86_64 platforms when the `multiversion_x86` feature is enabled.
///
/// # Performance (1024-bit, Ice Lake x86)
///
/// | Configuration | Speed |
/// |---------------|-------|
/// | Default (no SIMD) | ~10ns |
/// | `multiversion_x86` feature | ~3-4ns |
///
/// On ARM, NEON is the baseline so default builds are already fast.
///
/// For batch comparisons, see [`hamming_bitwise_slice_batch`].
///
/// # Panics
///
/// Panics if the two slices are not the same length.
#[cfg(all(
    not(feature = "multiversion_x86"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[inline(always)]
pub fn hamming_bitwise_slice(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);

    let main: u32 = a_chunks.clone().zip(b_chunks.clone())
        .map(|(a, b)| {
            let a = u64::from_ne_bytes(a.try_into().unwrap());
            let b = u64::from_ne_bytes(b.try_into().unwrap());
            (a ^ b).count_ones()
        })
        .sum();

    let rem: u32 = a_chunks.remainder().iter().zip(b_chunks.remainder())
        .map(|(a, b)| (a ^ b).count_ones())
        .sum();

    main + rem
}

/// Calculate the bitwise Hamming distance between two byte slices.
///
/// This function uses runtime CPU detection to dispatch to optimized SIMD implementations
/// on x86/x86_64 platforms when the `multiversion_x86` feature is enabled.
///
/// # Performance (1024-bit, Ice Lake x86)
///
/// | Configuration | Speed |
/// |---------------|-------|
/// | Default (no SIMD) | ~10ns |
/// | `multiversion_x86` feature | ~3-4ns |
///
/// On ARM, NEON is the baseline so default builds are already fast.
///
/// For batch comparisons, see [`hamming_bitwise_slice_batch`].
///
/// # Panics
///
/// Panics if the two slices are not the same length.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline(always)]
pub fn hamming_bitwise_slice(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Compute Hamming distance for fixed-size byte arrays.
///
/// Use this when the embedding size is known at compile time. The const generic `N`
/// represents the number of bytes.
///
/// Common sizes:
/// - `N=64`: 512-bit embedding
/// - `N=96`: 768-bit embedding
/// - `N=128`: 1024-bit embedding
/// - `N=256`: 2048-bit embedding
///
/// # Performance (1024-bit, Ice Lake x86)
///
/// | Configuration | Speed |
/// |---------------|-------|
/// | Default (no SIMD) | ~10ns |
/// | `multiversion_x86` feature | ~3ns |
///
/// For batch comparisons, see [`hamming_bitwise_array_batch`] which can achieve
/// ~0.8ns per comparison.
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
    "x86_64+avx512vpopcntdq+avx512vl+popcnt",
    "x86_64+avx512bw+avx512vl+popcnt",
    "x86_64+avx2+popcnt",
    "x86_64+sse4.2+popcnt",
    "x86+avx2+popcnt",
    "x86+sse4.2+popcnt",
))]
#[inline(always)]
pub fn hamming_bitwise_array<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);

    let main: u32 = a_chunks.clone().zip(b_chunks.clone())
        .map(|(a, b)| {
            let a = u64::from_ne_bytes(a.try_into().unwrap());
            let b = u64::from_ne_bytes(b.try_into().unwrap());
            (a ^ b).count_ones()
        })
        .sum();

    let rem: u32 = a_chunks.remainder().iter().zip(b_chunks.remainder())
        .map(|(a, b)| (a ^ b).count_ones())
        .sum();

    main + rem
}

/// Compute Hamming distance for fixed-size byte arrays.
///
/// Use this when the embedding size is known at compile time. The const generic `N`
/// represents the number of bytes.
///
/// Common sizes:
/// - `N=64`: 512-bit embedding
/// - `N=96`: 768-bit embedding
/// - `N=128`: 1024-bit embedding
/// - `N=256`: 2048-bit embedding
///
/// # Performance (1024-bit, Ice Lake x86)
///
/// | Configuration | Speed |
/// |---------------|-------|
/// | Default (no SIMD) | ~10ns |
/// | `multiversion_x86` feature | ~3ns |
///
/// For batch comparisons, see [`hamming_bitwise_array_batch`] which can achieve
/// ~0.8ns per comparison.
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
    not(feature = "multiversion_x86"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[inline(always)]
pub fn hamming_bitwise_array<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);

    let main: u32 = a_chunks.clone().zip(b_chunks.clone())
        .map(|(a, b)| {
            let a = u64::from_ne_bytes(a.try_into().unwrap());
            let b = u64::from_ne_bytes(b.try_into().unwrap());
            (a ^ b).count_ones()
        })
        .sum();

    let rem: u32 = a_chunks.remainder().iter().zip(b_chunks.remainder())
        .map(|(a, b)| (a ^ b).count_ones())
        .sum();

    main + rem
}

/// Compute Hamming distance for fixed-size byte arrays.
///
/// Use this when the embedding size is known at compile time. The const generic `N`
/// represents the number of bytes.
///
/// Common sizes:
/// - `N=64`: 512-bit embedding
/// - `N=96`: 768-bit embedding
/// - `N=128`: 1024-bit embedding
/// - `N=256`: 2048-bit embedding
///
/// # Performance (1024-bit, Ice Lake x86)
///
/// | Configuration | Speed |
/// |---------------|-------|
/// | Default (no SIMD) | ~10ns |
/// | `multiversion_x86` feature | ~3ns |
///
/// For batch comparisons, see [`hamming_bitwise_array_batch`] which can achieve
/// ~0.8ns per comparison.
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
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline(always)]
pub fn hamming_bitwise_array<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Compute Hamming distance from one source embedding to many targets.
///
/// This is significantly faster than calling [`hamming_bitwise_array`] in a loop (~4x faster
/// on Ice Lake x86 with `multiversion_x86`).
///
/// # Performance (1024-bit, Ice Lake x86 with `multiversion_x86`)
///
/// ~0.8ns per comparison (vs ~3ns for single comparisons in a loop).
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
/// use hamming_bitwise_fast::hamming_bitwise_array_batch;
///
/// let source: [u8; 128] = [0; 128];
/// let targets = vec![[1u8; 128], [2u8; 128], [3u8; 128]];
///
/// // Pre-allocate result vec once and reuse across calls for best performance
/// let mut distances = vec![0u32; 3];
///
/// hamming_bitwise_array_batch(&source, &targets, &mut distances);
/// ```
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
pub fn hamming_bitwise_array_batch<const N: usize>(
    source: &[u8; N],
    targets: &[[u8; N]],
    out: &mut [u32],
) {
    assert_eq!(targets.len(), out.len());

    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        let a_chunks = source.chunks_exact(8);
        let b_chunks = target.chunks_exact(8);

        let main: u32 = a_chunks.clone().zip(b_chunks.clone())
            .map(|(a, b)| {
                let a = u64::from_ne_bytes(a.try_into().unwrap());
                let b = u64::from_ne_bytes(b.try_into().unwrap());
                (a ^ b).count_ones()
            })
            .sum();

        let rem: u32 = a_chunks.remainder().iter().zip(b_chunks.remainder())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum();

        *dist = main + rem;
    }
}

/// Compute Hamming distance from one source embedding to many targets.
///
/// This is significantly faster than calling [`hamming_bitwise_array`] in a loop (~4x faster
/// on Ice Lake x86 with `multiversion_x86`).
///
/// # Performance (1024-bit, Ice Lake x86 with `multiversion_x86`)
///
/// ~0.8ns per comparison (vs ~3ns for single comparisons in a loop).
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
/// use hamming_bitwise_fast::hamming_bitwise_array_batch;
///
/// let source: [u8; 128] = [0; 128];
/// let targets = vec![[1u8; 128], [2u8; 128], [3u8; 128]];
///
/// // Pre-allocate result vec once and reuse across calls for best performance
/// let mut distances = vec![0u32; 3];
///
/// hamming_bitwise_array_batch(&source, &targets, &mut distances);
/// ```
#[cfg(all(
    not(feature = "multiversion_x86"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[inline(always)]
pub fn hamming_bitwise_array_batch<const N: usize>(
    source: &[u8; N],
    targets: &[[u8; N]],
    out: &mut [u32],
) {
    assert_eq!(targets.len(), out.len());

    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        let a_chunks = source.chunks_exact(8);
        let b_chunks = target.chunks_exact(8);

        let main: u32 = a_chunks.clone().zip(b_chunks.clone())
            .map(|(a, b)| {
                let a = u64::from_ne_bytes(a.try_into().unwrap());
                let b = u64::from_ne_bytes(b.try_into().unwrap());
                (a ^ b).count_ones()
            })
            .sum();

        let rem: u32 = a_chunks.remainder().iter().zip(b_chunks.remainder())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum();

        *dist = main + rem;
    }
}

/// Compute Hamming distance from one source embedding to many targets.
///
/// This is significantly faster than calling [`hamming_bitwise_array`] in a loop (~4x faster
/// on Ice Lake x86 with `multiversion_x86`).
///
/// # Performance (1024-bit, Ice Lake x86 with `multiversion_x86`)
///
/// ~0.8ns per comparison (vs ~3ns for single comparisons in a loop).
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
/// use hamming_bitwise_fast::hamming_bitwise_array_batch;
///
/// let source: [u8; 128] = [0; 128];
/// let targets = vec![[1u8; 128], [2u8; 128], [3u8; 128]];
///
/// // Pre-allocate result vec once and reuse across calls for best performance
/// let mut distances = vec![0u32; 3];
///
/// hamming_bitwise_array_batch(&source, &targets, &mut distances);
/// ```
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline(always)]
pub fn hamming_bitwise_array_batch<const N: usize>(
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
/// This is the slice-based equivalent of [`hamming_bitwise_array_batch`], useful when
/// embedding sizes are not known at compile time.
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
/// use hamming_bitwise_fast::hamming_bitwise_slice_batch;
///
/// let source = vec![0u8; 128];
/// let targets_owned: Vec<Vec<u8>> = vec![vec![1u8; 128], vec![2u8; 128], vec![3u8; 128]];
/// let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();
///
/// // Pre-allocate result vec once and reuse across calls for best performance
/// let mut distances = vec![0u32; 3];
///
/// hamming_bitwise_slice_batch(&source, &targets, &mut distances);
/// ```
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
pub fn hamming_bitwise_slice_batch(source: &[u8], targets: &[&[u8]], out: &mut [u32]) {
    assert_eq!(targets.len(), out.len());

    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        assert_eq!(source.len(), target.len());
        let a_chunks = source.chunks_exact(8);
        let b_chunks = target.chunks_exact(8);

        let main: u32 = a_chunks.clone().zip(b_chunks.clone())
            .map(|(a, b)| {
                let a = u64::from_ne_bytes(a.try_into().unwrap());
                let b = u64::from_ne_bytes(b.try_into().unwrap());
                (a ^ b).count_ones()
            })
            .sum();

        let rem: u32 = a_chunks.remainder().iter().zip(b_chunks.remainder())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum();

        *dist = main + rem;
    }
}

/// Compute Hamming distance from one source slice to many target slices.
///
/// This is the slice-based equivalent of [`hamming_bitwise_array_batch`], useful when
/// embedding sizes are not known at compile time.
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
/// use hamming_bitwise_fast::hamming_bitwise_slice_batch;
///
/// let source = vec![0u8; 128];
/// let targets_owned: Vec<Vec<u8>> = vec![vec![1u8; 128], vec![2u8; 128], vec![3u8; 128]];
/// let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();
///
/// // Pre-allocate result vec once and reuse across calls for best performance
/// let mut distances = vec![0u32; 3];
///
/// hamming_bitwise_slice_batch(&source, &targets, &mut distances);
/// ```
#[cfg(all(
    not(feature = "multiversion_x86"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[inline(always)]
pub fn hamming_bitwise_slice_batch(source: &[u8], targets: &[&[u8]], out: &mut [u32]) {
    assert_eq!(targets.len(), out.len());

    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        assert_eq!(source.len(), target.len());
        let a_chunks = source.chunks_exact(8);
        let b_chunks = target.chunks_exact(8);

        let main: u32 = a_chunks.clone().zip(b_chunks.clone())
            .map(|(a, b)| {
                let a = u64::from_ne_bytes(a.try_into().unwrap());
                let b = u64::from_ne_bytes(b.try_into().unwrap());
                (a ^ b).count_ones()
            })
            .sum();

        let rem: u32 = a_chunks.remainder().iter().zip(b_chunks.remainder())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum();

        *dist = main + rem;
    }
}

/// Compute Hamming distance from one source slice to many target slices.
///
/// This is the slice-based equivalent of [`hamming_bitwise_array_batch`], useful when
/// embedding sizes are not known at compile time.
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
/// use hamming_bitwise_fast::hamming_bitwise_slice_batch;
///
/// let source = vec![0u8; 128];
/// let targets_owned: Vec<Vec<u8>> = vec![vec![1u8; 128], vec![2u8; 128], vec![3u8; 128]];
/// let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();
///
/// // Pre-allocate result vec once and reuse across calls for best performance
/// let mut distances = vec![0u32; 3];
///
/// hamming_bitwise_slice_batch(&source, &targets, &mut distances);
/// ```
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline(always)]
pub fn hamming_bitwise_slice_batch(source: &[u8], targets: &[&[u8]], out: &mut [u32]) {
    assert_eq!(targets.len(), out.len());

    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        *dist = hamming_bitwise_slice(source, target);
    }
}

/// Deprecated: Use [`hamming_bitwise_slice`] instead, or consider
/// [`hamming_bitwise_array`] for fixed-size arrays under 2048 bits (10-100% faster) or
/// [`hamming_bitwise_array_batch`] for comparing one source against many targets.
#[deprecated(
    since = "1.1.0",
    note = "renamed to hamming_bitwise_slice; consider hamming_bitwise_array for fixed-size arrays under 2048 bits (10-100% faster) or hamming_bitwise_array_batch for bulk comparisons"
)]
#[inline(always)]
pub fn hamming_bitwise_fast(x: &[u8], y: &[u8]) -> u32 {
    hamming_bitwise_slice(x, y)
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
    fn hamming_bitwise_array_batch_correctness() {
        let source: [u8; 128] = [0; 128];
        let targets = vec![
            [0xFFu8; 128], // 1024 bits different
            [0u8; 128],    // 0 bits different
            [1u8; 128],    // 128 bits different (one bit per byte)
        ];
        let mut out = vec![0u32; 3];

        hamming_bitwise_array_batch(&source, &targets, &mut out);

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
        hamming_bitwise_array_batch(&source, &targets, &mut out);
        assert_eq!(out[0], 104);
        assert_eq!(out[1], 0);
    }

    #[test]
    fn hamming_bitwise_slice_batch_correctness() {
        let source = vec![0u8; 128];
        let targets_owned = vec![
            vec![0xFFu8; 128], // 1024 bits different
            vec![0u8; 128],    // 0 bits different
            vec![1u8; 128],    // 128 bits different (one bit per byte)
        ];
        let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();
        let mut out = vec![0u32; 3];

        hamming_bitwise_slice_batch(&source, &targets, &mut out);

        assert_eq!(out[0], 1024);
        assert_eq!(out[1], 0);
        assert_eq!(out[2], 128);

        // Verify results match individual slice calls
        for (i, target) in targets.iter().enumerate() {
            assert_eq!(out[i], hamming_bitwise_slice(&source, target));
        }
    }

    #[test]
    fn hamming_bitwise_slice_batch_empty() {
        let source = vec![0u8; 128];
        let targets: Vec<&[u8]> = vec![];
        let mut out: Vec<u32> = vec![];

        // Should succeed with empty inputs
        hamming_bitwise_slice_batch(&source, &targets, &mut out);
    }

    #[test]
    #[should_panic]
    fn hamming_bitwise_slice_batch_output_size_mismatch() {
        let source = vec![0u8; 128];
        let targets_owned = vec![vec![0u8; 128], vec![0u8; 128]];
        let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();
        let mut out = vec![0u32; 1]; // Wrong size!

        hamming_bitwise_slice_batch(&source, &targets, &mut out);
    }
}

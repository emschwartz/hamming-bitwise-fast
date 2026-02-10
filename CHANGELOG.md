# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-02-10

### Added

- `array` module with `distance<const N>(&[u8; N], &[u8; N])` and `batch<const N>` for
  fixed-size byte arrays. Recommended when the vector size is known at compile time.
- `slice` module with `distance(&[u8], &[u8])` and `batch` for variable-length slices.
- `batch` functions for fast one-to-many Hamming distance comparisons.
- Runtime CPU dispatch on x86 (AVX-512 VPOPCNTDQ / AVX2 / SSE4.2) via the
  `multiversion` crate, enabled by default with the `multiversion_x86` feature.
- Platform-specific `distance_impl`: u64-chunked processing on x86 for
  auto-vectorization, simple byte iteration on ARM (NEON-friendly).
- Zero-cost `asm!` barrier in `array::batch` to prevent LLVM from emitting
  slow AVX-512 gather instructions on contiguous array data.
- Comprehensive crate-level documentation with platform behavior table, usage
  examples, and feature flag guidance.
- `#[inline(always)]` on all public functions to enable cross-crate
  auto-vectorization when users enable LTO.

### Changed

- The original `hamming_bitwise_fast()` function is now a convenience alias
  for `slice::distance`. Existing callers are unaffected.

### Performance

- **Recommended: enable `lto = true` in your release profile.** On x86, this
  allows the compiler to auto-vectorize across the crate boundary, improving
  `array::distance` by up to 3.4x (e.g., 13.5ns → 3.9ns at 256 bytes on AMD
  EPYC Zen 4). Use full LTO (`true`), not thin (`"thin"`).

### Removed

- `naive_hamming_distance` and `naive_hamming_distance_iter` (were `#[doc(hidden)]`).

## [1.0.0] - 2024-12-20

Initial release with bitwise Hamming distance for byte slices using a
u64-chunked XOR + popcount algorithm amenable to auto-vectorization.

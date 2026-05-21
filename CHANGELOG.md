# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-05-20

### Added

- `array` module with `distance<const N>(&[u8; N], &[u8; N])` and `batch<const N>` for
  fixed-size byte arrays. Recommended when the vector size is known at compile time.
- `slice` module with `distance(&[u8], &[u8])` and `batch` for variable-length slices.
- `batch` functions for fast one-to-many Hamming distance comparisons.
- Runtime CPU dispatch on x86 (AVX-512 VPOPCNTDQ / AVX2 / SSE4.2) via the
  `multiversion` crate, enabled by default with the `multiversion_x86` feature.
- Platform-specific `distance_impl`: u64-chunked processing on x86 for
  auto-vectorization, simple byte iteration on ARM (NEON-friendly).
- `asm!` barrier in `array::batch` (zero-cost under LTO; load-bearing without
  LTO) that prevents LLVM from emitting cross-iteration VPGATHERQQ gather
  instructions on contiguous array data. With LTO + multiversion (the
  recommended config), LLVM doesn't emit gathers either way and the barrier
  is a verified no-op; without LTO, the barrier is ~4× faster than the
  un-barriered form. See the PERFORMANCE INVARIANT block in `src/array.rs`.
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
  `array::distance` by up to 3.1x on AMD EPYC Zen 5 (e.g., 6.1ns → 2.0ns at
  128 bytes). Use full LTO (`true`), not thin (`"thin"`).
- Benchmark tables in the README now cover AMD EPYC 9845 Zen 5 (AVX-512
  VPOPCNTDQ), AMD EPYC 7713 Zen 3 (AVX2 only), and Apple M2 Max, with
  paired LTO impact tables and a methodology note on cloud-VM variance.

### Removed

- `naive_hamming_distance` and `naive_hamming_distance_iter` (were `#[doc(hidden)]`).

## [1.0.0] - 2024-12-20

Initial release with bitwise Hamming distance for byte slices using a
u64-chunked XOR + popcount algorithm amenable to auto-vectorization.

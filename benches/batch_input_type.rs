//! Benchmarks comparing array batch vs slice batch performance, plus a
//! dedicated `gather_demo` group that A/B/Cs the three known ways to deal
//! with AVX-512 cross-iteration gather emission.
//!
//! # Background: AVX-512 Gather Avoidance
//!
//! When LLVM sees `targets: &[[u8; N]]` (contiguous array of arrays) and has
//! AVX-512 target features enabled, it can transform the outer loop into a
//! VPGATHERQQ-based form — reading one element from many targets per
//! instruction. On Zen 5 each such gather is ~2-10× slower than the
//! equivalent contiguous VMOVDQU64 + VPOPCNTQ form (separate memory fetches,
//! cache locality lost, no prefetcher).
//!
//! ## When LLVM actually emits gathers
//!
//! Whether the bad transformation happens depends on what LLVM can see:
//!
//! - **With LTO + multiversion (the recommended user config):** LLVM inlines
//!   across the multiversion dispatch boundary, sees `N` is a compile-time
//!   constant, unrolls the inner loop, and produces zero gathers regardless
//!   of what the source code looks like.
//! - **Without LTO + multiversion:** each multiversion specialization is a
//!   separate translation unit. LLVM can't see `N` and resorts to outer-loop
//!   vectorization via VPGATHERQQ. Measured: 112 such instructions in this
//!   benchmark binary, ~4× runtime slowdown.
//!
//! So the production asm! barrier in `array::batch` is load-bearing for
//! no-LTO users and a verified no-op for LTO users (assembly is identical
//! with and without it under LTO).
//!
//! ## The three variants in `gather_demo`
//!
//! - `no_blackbox_slow_gather` — no barrier. Hits VPGATHERQQ under no-LTO.
//! - `blackbox_fast_loads` — `std::hint::black_box`. Prevents the gather,
//!   but compiles to a stack store + reload (~5-cycle store-forwarding
//!   penalty per iteration). Under LTO + AVX-512 this is ~7× slower than
//!   the asm! barrier.
//! - `asm_barrier_zero_cost` — what the production code uses. The asm!
//!   barrier with `nomem` keeps the target pointer in a register and
//!   blocks LLVM's cross-iteration analysis, but at zero per-iteration cost.
//!
//! ## Verifying the barrier still works
//!
//! Build under no-LTO and disassemble:
//!
//! ```sh
//! CARGO_PROFILE_BENCH_LTO=false cargo bench --bench batch_input_type --no-run
//! objdump -d target/release/deps/batch_input_type-* | \
//!   awk '/<.*batch_with_asm_barrier.*>:/{p=1} p; /ret$/{p=0}' | \
//!   grep -c 'vpgather\|vpscatter'   # should be 0
//! ```
//!
//! Same disassembly against `batch_no_black_box` should produce a non-zero
//! count (currently 112 across the multiversion specializations).
//!
//! Run with: cargo bench --features multiversion_x86 --bench batch_input_type -- --quick

mod helpers;

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use hamming_bitwise_fast::{array, slice};
use helpers::{random_bytes, random_bytes_array, random_bytes_vec};

const BATCH: usize = 1000;

// ============================================================================
// Main benchmarks: comparing batch APIs
// ============================================================================

fn benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_input_type");

    macro_rules! bench_size {
        ($size:expr) => {{
            let bits = format!("{}b", $size * 8);

            // Array batch
            {
                let source: [u8; $size] = random_bytes();
                let targets: Vec<[u8; $size]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];

                group.bench_with_input(
                    BenchmarkId::new("array_batch", &bits),
                    &$size,
                    |bencher, _| {
                        bencher.iter(|| {
                            array::batch(black_box(&source), black_box(&targets), &mut out);
                            black_box(out[0])
                        })
                    },
                );

                // Array via slice batch (convert arrays to slices)
                let targets_refs: Vec<&[u8]> = targets.iter().map(|a| a.as_slice()).collect();
                group.bench_with_input(
                    BenchmarkId::new("array_via_slice_batch", &bits),
                    &$size,
                    |bencher, _| {
                        bencher.iter(|| {
                            slice::batch(
                                black_box(&source[..]),
                                black_box(&targets_refs),
                                &mut out,
                            );
                            black_box(out[0])
                        })
                    },
                );

                // Array loop (call single function repeatedly)
                group.bench_with_input(
                    BenchmarkId::new("array_loop_single", &bits),
                    &$size,
                    |bencher, _| {
                        bencher.iter(|| {
                            for (target, dist) in black_box(&targets).iter().zip(out.iter_mut()) {
                                *dist = array::distance(black_box(&source), target);
                            }
                            black_box(out[0])
                        })
                    },
                );
            }

            // Slice batch
            {
                let source = random_bytes_vec($size);
                let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec($size)).collect();
                let targets_refs: Vec<&[u8]> = targets.iter().map(|v| v.as_slice()).collect();
                let mut out = vec![0u32; BATCH];

                group.bench_with_input(
                    BenchmarkId::new("slice_batch", &bits),
                    &$size,
                    |bencher, _| {
                        bencher.iter(|| {
                            slice::batch(black_box(&source), black_box(&targets_refs), &mut out);
                            black_box(out[0])
                        })
                    },
                );

                // Slice loop (call single function repeatedly)
                group.bench_with_input(
                    BenchmarkId::new("slice_loop_single", &bits),
                    &$size,
                    |bencher, _| {
                        bencher.iter(|| {
                            for (target, dist) in black_box(&targets).iter().zip(out.iter_mut()) {
                                *dist = slice::distance(black_box(&source), target);
                            }
                            black_box(out[0])
                        })
                    },
                );
            }
        }};
    }

    bench_size!(64);
    bench_size!(128);
    bench_size!(256);

    group.finish();
}

// ============================================================================
// Demonstration: gather avoidance strategies on x86 AVX-512
// Three approaches: no barrier (gathers), black_box, asm! barrier
// (Only meaningful with --features multiversion_x86)
// ============================================================================

#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
mod gather_demo {
    use super::*;

    /// Inner function WITHOUT black_box - LLVM generates slow gather instructions.
    #[multiversion::multiversion(targets(
        "x86_64+avx512vpopcntdq+avx512vl+popcnt",
        "x86_64+avx512bw+avx512vl+popcnt",
        "x86_64+avx2+popcnt",
        "x86_64+sse4.2+popcnt",
    ))]
    fn batch_no_black_box<const N: usize>(source: &[u8; N], targets: &[[u8; N]], out: &mut [u32]) {
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            // No black_box - compiler can see across iterations and use gather
            let a_chunks = source.chunks_exact(8);
            let b_chunks = target.chunks_exact(8);

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

            *dist = main + rem;
        }
    }

    /// Inner function WITH black_box - prevents gather but has store-forwarding penalty.
    #[multiversion::multiversion(targets(
        "x86_64+avx512vpopcntdq+avx512vl+popcnt",
        "x86_64+avx512bw+avx512vl+popcnt",
        "x86_64+avx2+popcnt",
        "x86_64+sse4.2+popcnt",
    ))]
    fn batch_with_black_box<const N: usize>(
        source: &[u8; N],
        targets: &[[u8; N]],
        out: &mut [u32],
    ) {
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            // black_box hides target from cross-iteration optimization
            // but compiles to store+reload (~5 cycle store-forwarding penalty)
            let target: &[u8] = std::hint::black_box(target);

            let a_chunks = source.chunks_exact(8);
            let b_chunks = target.chunks_exact(8);

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

            *dist = main + rem;
        }
    }

    /// Make a pointer opaque to LLVM without store-forwarding penalty.
    /// Identical to the production `opaque_ptr` in `src/array.rs`.
    #[inline(always)]
    unsafe fn opaque_ptr<T>(mut ptr: *const T) -> *const T {
        core::arch::asm!("/* {0} */", inout(reg) ptr, options(nomem, nostack, preserves_flags));
        ptr
    }

    /// Inner function WITH asm! barrier - prevents gather at zero cost.
    #[multiversion::multiversion(targets(
        "x86_64+avx512vpopcntdq+avx512vl+popcnt",
        "x86_64+avx512bw+avx512vl+popcnt",
        "x86_64+avx2+popcnt",
        "x86_64+sse4.2+popcnt",
    ))]
    fn batch_with_asm_barrier<const N: usize>(
        source: &[u8; N],
        targets: &[[u8; N]],
        out: &mut [u32],
    ) {
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            // asm! barrier: pointer stays in register (no store-forwarding)
            // but LLVM can't analyze stride patterns through the sideeffect
            let target = unsafe { &*opaque_ptr(target as *const [u8; N]) };

            let a_chunks = source.chunks_exact(8);
            let b_chunks = target.chunks_exact(8);

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

            *dist = main + rem;
        }
    }

    pub fn benchmarks(c: &mut Criterion) {
        let mut group = c.benchmark_group("gather_demo");

        macro_rules! bench_size {
            ($size:expr) => {{
                let bits = format!("{}b", $size * 8);
                let source: [u8; $size] = random_bytes();
                let targets: Vec<[u8; $size]> = random_bytes_array(BATCH);
                let mut out = vec![0u32; BATCH];

                group.bench_with_input(
                    BenchmarkId::new("no_blackbox_slow_gather", &bits),
                    &$size,
                    |bencher, _| {
                        bencher.iter(|| {
                            batch_no_black_box(black_box(&source), black_box(&targets), &mut out);
                            black_box(out[0])
                        })
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("blackbox_fast_loads", &bits),
                    &$size,
                    |bencher, _| {
                        bencher.iter(|| {
                            batch_with_black_box(black_box(&source), black_box(&targets), &mut out);
                            black_box(out[0])
                        })
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("asm_barrier_zero_cost", &bits),
                    &$size,
                    |bencher, _| {
                        bencher.iter(|| {
                            batch_with_asm_barrier(
                                black_box(&source),
                                black_box(&targets),
                                &mut out,
                            );
                            black_box(out[0])
                        })
                    },
                );
            }};
        }

        bench_size!(64);
        bench_size!(128);
        bench_size!(256);

        group.finish();
    }
}

#[cfg(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
))]
criterion_group!(benches, benchmarks, gather_demo::benchmarks);

#[cfg(not(all(
    feature = "multiversion_x86",
    any(target_arch = "x86", target_arch = "x86_64")
)))]
criterion_group!(benches, benchmarks);

criterion_main!(benches);

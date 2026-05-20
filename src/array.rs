//! Fixed-size array APIs for bitwise Hamming distance.
//!
//! Use this module when the vector size is known at compile time (e.g., 1024-bit
//! embeddings stored as `[u8; 128]`). This is faster than the equivalent
//! [`slice`](crate::slice) API.

// ============================================================================
// PERFORMANCE INVARIANT: AVX-512 Gather Avoidance (load-bearing without LTO)
// ============================================================================
//
// `batch()` iterates over `&[[u8; N]]` — contiguous memory. With AVX-512
// target features available, LLVM can transform that loop to use VPGATHERQQ
// gather instructions, which are 2–10x slower than contiguous VMOVDQU64 loads
// (each element fetched separately, cache locality destroyed, no prefetcher
// win). The asm! barrier on `target` below forces LLVM to use the simple
// load-xor-popcount form instead.
//
// Whether the barrier matters depends on LTO:
//
//   - With LTO + multiversion (the recommended config): LLVM inlines across
//     the multiversion dispatch boundary, sees `N` is a compile-time constant,
//     unrolls the inner loop, and never emits gathers in the first place.
//     The barrier is a verified no-op here — assembly is identical with and
//     without it.
//
//   - Without LTO + multiversion: each multiversion specialization is a
//     separate translation unit. LLVM can't see N, falls back to outer-loop
//     vectorization, and emits VPGATHERQQ across iterations. Measured: 112
//     such instructions per benchmark binary, ~4x slower than the barriered
//     form on Zen 5. The barrier is the difference between fast and slow here.
//
// The barrier is kept unconditionally as defense for users who don't enable
// LTO (and as insurance against future LLVM versions changing the heuristic
// under LTO too). It has no measurable cost under LTO.
//
// Why not `black_box`? Both prevent the gather, but `black_box` compiles to
// a stack store + reload (~5-cycle store-forwarding penalty per iteration).
// Under LTO + AVX-512 that penalty is ~7x slower than the asm! barrier
// (gather_demo: black_box = 2.85µs vs asm_barrier = 410ns at 64B).
//
// On non-x86 (ARM etc.), no barrier is needed — gather instructions don't
// exist on those architectures, and `opaque_ptr` is a plain identity.
//
// Verify: inspect AVX-512 assembly under CARGO_PROFILE_BENCH_LTO=false for
//         absence of VPGATHERQQ in the asm-barriered batch loop.
// Proof:  benches/batch_input_type.rs `gather_demo` (no_barrier / black_box /
//         asm_barrier A/B/C comparison).
// ============================================================================

/// Make a pointer opaque to LLVM's stride analysis without store-forwarding.
///
/// On x86, uses `asm!` with `nomem` + `nostack` — the pointer stays in a
/// register but LLVM treats it as a new, unknown value (preventing the
/// outer-loop gather vectorization LLVM otherwise picks under no-LTO
/// multiversion builds). On non-x86, returns the pointer unchanged since
/// gather instructions don't exist on those architectures.
///
/// # Safety
///
/// The pointer must be valid. The asm block is a no-op (empty template),
/// so the returned pointer is identical to the input.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
#[allow(clippy::pointers_in_nomem_asm_block)]
unsafe fn opaque_ptr<T>(mut ptr: *const T) -> *const T {
    core::arch::asm!("/* {0} */", inout(reg) ptr, options(nomem, nostack, preserves_flags));
    ptr
}

// ============================================================================
// Public API
// ============================================================================

/// Compute the bitwise Hamming distance between two fixed-size byte arrays.
///
/// This is the recommended API when the vector size is known at compile time.
/// It is faster than [`slice::distance`](crate::slice::distance).
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
    all(
        feature = "multiversion_x86",
        any(target_arch = "x86", target_arch = "x86_64")
    ),
    multiversion::multiversion(targets(
        "x86_64+avx512vpopcntdq+avx512vl+popcnt",
        "x86_64+avx512bw+avx512vl+popcnt",
        "x86_64+avx2+popcnt",
        "x86_64+sse4.2+popcnt",
        "x86+avx2+popcnt",
        "x86+sse4.2+popcnt",
    ))
)]
#[inline(always)]
pub fn distance<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    crate::distance_impl(a, b)
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
    all(
        feature = "multiversion_x86",
        any(target_arch = "x86", target_arch = "x86_64")
    ),
    multiversion::multiversion(targets(
        "x86_64+avx512vpopcntdq+avx512vl+popcnt",
        "x86_64+avx512bw+avx512vl+popcnt",
        "x86_64+avx2+popcnt",
        "x86_64+sse4.2+popcnt",
        "x86+avx2+popcnt",
        "x86+sse4.2+popcnt",
    ))
)]
#[inline(always)]
pub fn batch<const N: usize>(source: &[u8; N], targets: &[[u8; N]], out: &mut [u32]) {
    assert_eq!(targets.len(), out.len());

    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        // Gather avoidance for no-LTO multiversion builds. With LTO this line
        // is a verified no-op; without LTO it prevents LLVM from emitting
        // VPGATHERQQ across iterations (~4x slowdown). See the module-level
        // PERFORMANCE INVARIANT block.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let target = unsafe { &*opaque_ptr(target as *const [u8; N]) };
        *dist = crate::distance_impl(source, target);
    }
}

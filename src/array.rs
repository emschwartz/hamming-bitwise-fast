//! Fixed-size array APIs for bitwise Hamming distance.
//!
//! Use this module when the vector size is known at compile time (e.g., 1024-bit
//! embeddings stored as `[u8; 128]`). This is faster than the equivalent
//! [`slice`](crate::slice) API.

// ============================================================================
// PERFORMANCE INVARIANT: AVX-512 Gather Avoidance
// ============================================================================
//
// batch() iterates over &[[u8; N]] — contiguous memory.
// On x86 AVX-512, LLVM can analyze stride patterns across iterations and emit
// VPGATHERQQ gather instructions. These are 2-10x SLOWER than contiguous
// VMOVDQU64 loads because each element requires a separate memory fetch.
//
// Mitigation: make the target pointer opaque to LLVM's stride analysis using
// an inline asm barrier. On x86, this uses `asm!("", inout(reg) ptr)` with
// `nomem` — the pointer stays in a register (no store-forwarding penalty) but
// LLVM's ScalarEvolution can't analyze through the `sideeffect` flag.
// On non-x86 (ARM etc.), no barrier is needed since gathers don't exist.
//
// This invariant MUST be maintained when modifying batch functions.
// Verify: inspect x86 AVX-512 assembly for absence of VPGATHERQQ.
// Proof: benches/batch_input_type.rs gather_demo (A/B comparison).
// ============================================================================

/// Make a pointer opaque to LLVM's stride analysis without store-forwarding.
///
/// On x86, uses `asm!` with `nomem` + `nostack` — the pointer stays in a
/// register but LLVM treats it as a new, unknown value (preventing gather
/// vectorization). On non-x86, returns the pointer unchanged since gather
/// instructions don't exist on those architectures.
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
        // PERFORMANCE INVARIANT: gather avoidance on x86 AVX-512
        //
        // Makes the target pointer opaque to prevent LLVM from emitting
        // VPGATHERQQ instructions from the contiguous &[[u8; N]] layout.
        // Uses asm! with nomem — pointer stays in register (no store-forwarding
        // penalty). See module-level comment and benches/batch_input_type.rs.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let target = unsafe { &*opaque_ptr(target as *const [u8; N]) };
        *dist = crate::distance_impl(source, target);
    }
}

//! Q3: Which SIMD instructions are most beneficial? How to target them effectively?
//!
//! Key questions:
//! - On ARM: NEON is great, but is it used by default?
//! - On x86: vectorized POPCNT (AVX-512 VPOPCNT) is massive - how to enable it?
//! - Which dispatch strategy works best: multiversion, pulp, or RUSTFLAGS?
//!
//! How to test different compiler optimizations:
//! ```sh
//! # Default (baseline)
//! cargo bench --bench q3_simd_dispatch
//!
//! # With target-cpu=native (uses all CPU features)
//! RUSTFLAGS="-C target-cpu=native" cargo bench --bench q3_simd_dispatch
//!
//! # With multiversion feature (runtime dispatch)
//! cargo bench --bench q3_simd_dispatch --features multiversion
//! ```
//!
//! Run with: cargo bench --bench q3_simd_dispatch

mod helpers;

use helpers::*;

fn main() {
    divan::main();
}

// ============================================================================
// Auto-vectorized (baseline): What the compiler does by default
// This is affected by RUSTFLAGS="-C target-cpu=native"
// ============================================================================

#[divan::bench_group]
mod auto_vectorized {
    use super::*;

    #[divan::bench(consts = [8, 12, 16, 32])]
    fn u64_iter<const N: usize>(bencher: divan::Bencher) {
        let a: Embedding<N> = random_embedding();
        let b: Embedding<N> = random_embedding();

        bencher.bench_local(|| {
            hamming_u64_iter(divan::black_box(&a), divan::black_box(&b))
        });
    }

    #[divan::bench(consts = [64, 96, 128, 256])]
    fn u8_iter<const N: usize>(bencher: divan::Bencher) {
        let a: [u8; N] = random_bytes();
        let b: [u8; N] = random_bytes();

        bencher.bench_local(|| {
            hamming_u8_iter(divan::black_box(&a), divan::black_box(&b))
        });
    }
}

// ============================================================================
// Multiversion: Runtime CPU feature detection via the multiversion crate
// Only active when compiled with --features multiversion
// ============================================================================

#[cfg(feature = "multiversion")]
#[divan::bench_group]
mod multiversion_dispatch {
    use super::*;

    #[divan::bench(consts = [8, 12, 16, 32])]
    fn u64<const N: usize>(bencher: divan::Bencher) {
        let a: Embedding<N> = random_embedding();
        let b: Embedding<N> = random_embedding();

        bencher.bench_local(|| {
            hamming_multiversion(divan::black_box(&a), divan::black_box(&b))
        });
    }
}

// ============================================================================
// Pulp: Portable SIMD via the pulp crate
// This uses runtime detection but different dispatch mechanism than multiversion
// ============================================================================

#[divan::bench_group]
mod pulp_dispatch {
    use super::*;

    #[divan::bench(consts = [8, 12, 16, 32])]
    fn u64<const N: usize>(bencher: divan::Bencher) {
        let a: Embedding<N> = random_embedding();
        let b: Embedding<N> = random_embedding();

        bencher.bench_local(|| {
            hamming_pulp(divan::black_box(&a), divan::black_box(&b))
        });
    }
}

// ============================================================================
// Library's hamming<N> function: Uses internal platform-specific optimization
// ============================================================================

#[divan::bench_group]
mod library {
    use super::*;

    #[divan::bench(consts = [8, 12, 16, 32])]
    fn hamming<const N: usize>(bencher: divan::Bencher) {
        let a: Embedding<N> = random_embedding();
        let b: Embedding<N> = random_embedding();

        bencher.bench_local(|| {
            hamming_bitwise_fast::hamming(divan::black_box(&a), divan::black_box(&b))
        });
    }
}

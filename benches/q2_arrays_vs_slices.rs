//! Q2: Are arrays better than slices? How much does it matter?
//!
//! Key questions:
//! - Does the compiler optimize fixed-size arrays better?
//! - Can stepping by chunks (u64) negate the slice overhead?
//! - Does asserting the slice length is a multiple of 8 help?
//!
//! Run with: cargo bench --bench q2_arrays_vs_slices
//! Filter by size: cargo bench --bench q2_arrays_vs_slices -- 1024

mod helpers;

use hamming_bitwise_fast::hamming_bitwise_fast;
use helpers::*;

fn main() {
    divan::main();
}

// ============================================================================
// Fixed-size arrays: Compiler knows the exact size at compile time
// ============================================================================

#[divan::bench_group]
mod fixed_arrays {
    use super::*;

    #[divan::bench(consts = [64, 96, 128, 256], name = "u8_array")]
    fn u8_array<const N: usize>(bencher: divan::Bencher) {
        let a: [u8; N] = random_bytes();
        let b: [u8; N] = random_bytes();

        bencher.bench_local(|| {
            hamming_u8_iter(divan::black_box(&a), divan::black_box(&b))
        });
    }

    #[divan::bench(consts = [8, 12, 16, 32], name = "u64_array")]
    fn u64_array<const N: usize>(bencher: divan::Bencher) {
        let a: Embedding<N> = random_embedding();
        let b: Embedding<N> = random_embedding();

        bencher.bench_local(|| {
            hamming_u64_iter(divan::black_box(&a), divan::black_box(&b))
        });
    }
}

// ============================================================================
// Slices: Compiler doesn't know size at compile time
// ============================================================================

#[divan::bench_group]
mod slices {
    use super::*;

    /// Basic slice iteration (byte-by-byte)
    #[divan::bench(args = [64, 96, 128, 256], name = "slice_basic")]
    fn slice_basic(bencher: divan::Bencher, size: usize) {
        let a = random_bytes_vec(size);
        let b = random_bytes_vec(size);

        bencher.bench_local(|| {
            hamming_slice(divan::black_box(&a), divan::black_box(&b))
        });
    }

    /// Slice with assertion that length is multiple of 8 (does it help SIMD?)
    #[divan::bench(args = [64, 96, 128, 256], name = "slice_assert_mult8")]
    fn slice_assert_mult8(bencher: divan::Bencher, size: usize) {
        let a = random_bytes_vec(size);
        let b = random_bytes_vec(size);

        bencher.bench_local(|| {
            hamming_slice_assert_multiple8(divan::black_box(&a), divan::black_box(&b))
        });
    }

    /// Slice processed as u64 chunks (mimic what fixed u64 array does)
    #[divan::bench(args = [64, 96, 128, 256], name = "slice_u64_chunks")]
    fn slice_u64_chunks(bencher: divan::Bencher, size: usize) {
        let a = random_bytes_vec(size);
        let b = random_bytes_vec(size);

        bencher.bench_local(|| {
            hamming_slice_u64_chunks(divan::black_box(&a), divan::black_box(&b))
        });
    }

    /// Library's hamming_bitwise_fast (slice API)
    #[divan::bench(args = [64, 96, 128, 256], name = "library_slice")]
    fn library_slice_api(bencher: divan::Bencher, size: usize) {
        let a = random_bytes_vec(size);
        let b = random_bytes_vec(size);

        bencher.bench_local(|| {
            hamming_bitwise_fast(divan::black_box(&a), divan::black_box(&b))
        });
    }
}

// ============================================================================
// Library comparison: slice API vs array API
// ============================================================================

#[divan::bench_group]
mod library_apis {
    use super::*;

    /// Library's slice-based API (hamming_bitwise_fast)
    #[divan::bench(args = [64, 96, 128, 256], name = "slice_api")]
    fn slice_api(bencher: divan::Bencher, size: usize) {
        let a = random_bytes_vec(size);
        let b = random_bytes_vec(size);

        bencher.bench_local(|| {
            hamming_bitwise_fast(divan::black_box(&a), divan::black_box(&b))
        });
    }

    /// Library's const-generic array API (hamming<N>)
    #[divan::bench(consts = [8, 12, 16, 32], name = "array_api")]
    fn array_api<const N: usize>(bencher: divan::Bencher) {
        let a: Embedding<N> = random_embedding();
        let b: Embedding<N> = random_embedding();

        bencher.bench_local(|| {
            hamming_bitwise_fast::hamming(divan::black_box(&a), divan::black_box(&b))
        });
    }
}

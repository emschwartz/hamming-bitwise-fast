//! Q2: Are arrays better than slices? How much does it matter?
//!
//! Key questions:
//! - Does the compiler optimize fixed-size arrays better?
//! - Can processing as u64 chunks negate the slice overhead?
//! - Does asserting the slice length is a multiple of 8 help?
//!
//! Run with: cargo bench --bench q2_arrays_vs_slices
//! Filter by size: cargo bench --bench q2_arrays_vs_slices -- 1024

mod helpers;

use criterion::{black_box, criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration};
use hamming_bitwise_fast::{hamming_bitwise_array, hamming_bitwise_slice};
use helpers::*;

fn arrays_vs_slices(c: &mut Criterion) {
    let mut group = c.benchmark_group("arrays_vs_slices");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Linear));

    // ========================================================================
    // Fixed-size arrays: Compiler knows the exact size at compile time
    // ========================================================================
    macro_rules! bench_array {
        ($($bits:literal => $bytes:literal),+ $(,)?) => {
            $(
                {
                    let a: [u8; $bytes] = random_bytes();
                    let b: [u8; $bytes] = random_bytes();
                    group.bench_function(
                        BenchmarkId::new("array_u8_iter", concat!(stringify!($bits), "b")),
                        |bench| {
                            bench.iter(|| hamming_u8_iter(black_box(&a), black_box(&b)));
                        },
                    );
                }
            )+
        };
    }
    bench_array!(512 => 64, 768 => 96, 1024 => 128, 2048 => 256);

    // ========================================================================
    // Slices: Compiler doesn't know size at compile time
    // ========================================================================
    for size in BIT_SIZES {
        let bytes = size.bytes();
        let a = random_bytes_vec(bytes);
        let b = random_bytes_vec(bytes);

        // Basic slice iteration (byte-by-byte)
        group.bench_with_input(BenchmarkId::new("slice_basic", size), &size, |bench, _| {
            bench.iter(|| hamming_slice(black_box(&a), black_box(&b)));
        });

        // Slice with assertion that length is multiple of 8
        group.bench_with_input(
            BenchmarkId::new("slice_assert_mult8", size),
            &size,
            |bench, _| {
                bench.iter(|| hamming_slice_assert_multiple8(black_box(&a), black_box(&b)));
            },
        );

        // Slice processed as u64 chunks
        group.bench_with_input(
            BenchmarkId::new("slice_u64_chunks", size),
            &size,
            |bench, _| {
                bench.iter(|| hamming_slice_u64_chunks(black_box(&a), black_box(&b)));
            },
        );

        // Library's hamming_bitwise_fast (slice API)
        group.bench_with_input(
            BenchmarkId::new("hamming_bitwise_slice", size),
            &size,
            |bench, _| {
                bench.iter(|| hamming_bitwise_slice(black_box(&a), black_box(&b)));
            },
        );
    }

    // ========================================================================
    // Library's const-generic array API (hamming<N>)
    // ========================================================================
    macro_rules! bench_library_array {
        ($($bits:literal => $bytes:literal),+ $(,)?) => {
            $(
                {
                    let a: [u8; $bytes] = random_bytes();
                    let b: [u8; $bytes] = random_bytes();
                    group.bench_function(
                        BenchmarkId::new("hamming_bitwise_array", concat!(stringify!($bits), "b")),
                        |bench| {
                            bench.iter(|| hamming_bitwise_array(black_box(&a), black_box(&b)));
                        },
                    );
                }
            )+
        };
    }
    bench_library_array!(512 => 64, 768 => 96, 1024 => 128, 2048 => 256);

    group.finish();
}

criterion_group!(benches, arrays_vs_slices);
criterion_main!(benches);

//! Q6: Does multiversion improve slice performance?
//!
//! Key questions:
//! - Does runtime CPU dispatch benefit slice operations?
//! - How does multiversion slice compare to the array version?
//! - Is the dispatch overhead worth it for slices?
//!
//! Run with: cargo bench --bench q6_slice_multiversion --features multiversion_x86
//! Filter by size: cargo bench --bench q6_slice_multiversion -- 1024

mod helpers;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hamming_bitwise_fast::{hamming_bitwise_array, hamming_bitwise_slice};
use helpers::*;

fn slice_multiversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("slice_multiversion");

    for size in BIT_SIZES {
        let bytes = size.bytes();
        let a = random_bytes_vec(bytes);
        let b = random_bytes_vec(bytes);

        // Original v1 slice implementation (no multiversion, u64 chunked)
        group.bench_with_input(
            BenchmarkId::new("hamming_bitwise_slice_v1", size),
            &size,
            |bench, _| {
                bench.iter(|| hamming_bitwise_slice_v1(black_box(&a), black_box(&b)));
            },
        );

        // Current slice (multiversion when feature enabled)
        group.bench_with_input(
            BenchmarkId::new("hamming_bitwise_slice", size),
            &size,
            |bench, _| {
                bench.iter(|| hamming_bitwise_slice(black_box(&a), black_box(&b)));
            },
        );
    }

    // Also compare against array version for reference
    macro_rules! bench_array {
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
    bench_array!(512 => 64, 768 => 96, 1024 => 128, 2048 => 256);

    group.finish();
}

criterion_group!(benches, slice_multiversion);
criterion_main!(benches);

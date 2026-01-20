//! Q3: Which SIMD instructions are most beneficial? How to target them effectively?
//!
//! Key questions:
//! - On ARM: NEON is great, but is it used by default?
//! - On x86: vectorized POPCNT (AVX-512 VPOPCNT) is massive - how to enable it?
//! - Which dispatch strategy works best: multiversion or RUSTFLAGS?
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

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hamming_bitwise_fast::hamming_bitwise_array;
use helpers::*;

fn simd_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_dispatch");

    // ========================================================================
    // Auto-vectorized (baseline): What the compiler does by default
    // This is affected by RUSTFLAGS="-C target-cpu=native"
    // ========================================================================
    macro_rules! bench_u8_iter {
        ($($bits:literal => $bytes:literal),+ $(,)?) => {
            $(
                {
                    let a: [u8; $bytes] = random_bytes();
                    let b: [u8; $bytes] = random_bytes();
                    group.bench_function(
                        BenchmarkId::new("auto_u8_iter", concat!(stringify!($bits), "b")),
                        |bench| {
                            bench.iter(|| hamming_u8_iter(black_box(&a), black_box(&b)));
                        },
                    );
                }
            )+
        };
    }
    bench_u8_iter!(512 => 64, 768 => 96, 1024 => 128, 2048 => 256);

    macro_rules! bench_u8_chunks {
        ($($bits:literal => $bytes:literal),+ $(,)?) => {
            $(
                {
                    let a: [u8; $bytes] = random_bytes();
                    let b: [u8; $bytes] = random_bytes();
                    group.bench_function(
                        BenchmarkId::new("auto_u8_chunks", concat!(stringify!($bits), "b")),
                        |bench| {
                            bench.iter(|| hamming_u8_chunks(black_box(&a), black_box(&b)));
                        },
                    );
                }
            )+
        };
    }
    bench_u8_chunks!(512 => 64, 768 => 96, 1024 => 128, 2048 => 256);

    // ========================================================================
    // Multiversion: Runtime CPU feature detection via the multiversion crate
    // Only active when compiled with --features multiversion_x86
    // ========================================================================
    #[cfg(feature = "multiversion_x86")]
    {
        macro_rules! bench_multiversion {
            ($($bits:literal => $bytes:literal),+ $(,)?) => {
                $(
                    {
                        let a: [u8; $bytes] = random_bytes();
                        let b: [u8; $bytes] = random_bytes();
                        group.bench_function(
                            BenchmarkId::new("multiversion", concat!(stringify!($bits), "b")),
                            |bench| {
                                bench.iter(|| hamming_multiversion(black_box(&a), black_box(&b)));
                            },
                        );
                    }
                )+
            };
        }
        bench_multiversion!(512 => 64, 768 => 96, 1024 => 128, 2048 => 256);
    }

    // ========================================================================
    // Library's hamming<N> function: Uses internal platform-specific optimization
    // ========================================================================
    macro_rules! bench_library {
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
    bench_library!(512 => 64, 768 => 96, 1024 => 128, 2048 => 256);

    group.finish();
}

criterion_group!(benches, simd_dispatch);
criterion_main!(benches);

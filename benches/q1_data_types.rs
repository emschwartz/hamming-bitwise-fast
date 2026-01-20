//! Q1: What's the fastest way to compute Hamming distance on u8 arrays?
//!
//! Key questions:
//! - Is byte-by-byte iteration faster or slower than chunks_exact(8)?
//! - Does the remainder handling in chunks_exact add overhead?
//! - How does the library's hamming<N> compare?
//!
//! Run with: cargo bench --bench q1_data_types
//! With multiversion: cargo bench --features multiversion --bench q1_data_types

mod helpers;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hamming_bitwise_fast::hamming;
use helpers::*;

fn data_type_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_types");

    // ========================================================================
    // u8 array: byte-by-byte iteration
    // ========================================================================
    macro_rules! bench_u8_iter {
        ($($bits:literal => $bytes:literal),+ $(,)?) => {
            $(
                {
                    let a: [u8; $bytes] = random_bytes();
                    let b: [u8; $bytes] = random_bytes();
                    group.bench_function(
                        BenchmarkId::new("u8_iter", concat!(stringify!($bits), "b")),
                        |bench| {
                            bench.iter(|| hamming_u8_iter(black_box(&a), black_box(&b)));
                        },
                    );
                }
            )+
        };
    }
    bench_u8_iter!(512 => 64, 768 => 96, 1024 => 128, 2048 => 256);

    // ========================================================================
    // u8 array: chunks_exact(8) - processes as u64 without remainder
    // ========================================================================
    macro_rules! bench_u8_chunks {
        ($($bits:literal => $bytes:literal),+ $(,)?) => {
            $(
                {
                    let a: [u8; $bytes] = random_bytes();
                    let b: [u8; $bytes] = random_bytes();
                    group.bench_function(
                        BenchmarkId::new("u8_chunks", concat!(stringify!($bits), "b")),
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
    // u8 array: chunks_exact(8) with remainder handling
    // ========================================================================
    macro_rules! bench_u8_chunks_remainder {
        ($($bits:literal => $bytes:literal),+ $(,)?) => {
            $(
                {
                    let a: [u8; $bytes] = random_bytes();
                    let b: [u8; $bytes] = random_bytes();
                    group.bench_function(
                        BenchmarkId::new("u8_chunks_rem", concat!(stringify!($bits), "b")),
                        |bench| {
                            bench.iter(|| hamming_u8_chunks_with_remainder(black_box(&a), black_box(&b)));
                        },
                    );
                }
            )+
        };
    }
    bench_u8_chunks_remainder!(512 => 64, 768 => 96, 1024 => 128, 2048 => 256);

    // ========================================================================
    // Library's hamming<N> function
    // ========================================================================
    macro_rules! bench_library {
        ($($bits:literal => $bytes:literal),+ $(,)?) => {
            $(
                {
                    let a: [u8; $bytes] = random_bytes();
                    let b: [u8; $bytes] = random_bytes();
                    group.bench_function(
                        BenchmarkId::new("library_hamming", concat!(stringify!($bits), "b")),
                        |bench| {
                            bench.iter(|| hamming(black_box(&a), black_box(&b)));
                        },
                    );
                }
            )+
        };
    }
    bench_library!(512 => 64, 768 => 96, 1024 => 128, 2048 => 256);

    group.finish();
}

criterion_group!(benches, data_type_benchmarks);
criterion_main!(benches);

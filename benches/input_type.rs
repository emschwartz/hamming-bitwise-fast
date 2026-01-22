//! Compares fixed-size arrays vs dynamic slices.
//!
//! Key insight: Fixed-size arrays allow the compiler to unroll loops and
//! optimize bounds checks away, while slices require runtime length checks.
//!
//! Run with: cargo bench --bench input_type
//! Quick mode: cargo bench --bench input_type -- --quick

mod helpers;

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use helpers::{random_bytes, random_bytes_vec};

// ============================================================================
// Implementations
// ============================================================================

/// Fixed-size array: u64 chunks with remainder.
#[inline]
fn hamming_array<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);

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

    main + rem
}

/// Dynamic slice: u64 chunks with remainder.
#[inline]
fn hamming_slice(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);

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

    main + rem
}

// ============================================================================
// Benchmarks
// ============================================================================

fn array_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("input_type/fixed_array");

    macro_rules! bench_size {
        ($size:expr) => {{
            let a: [u8; $size] = random_bytes();
            let b: [u8; $size] = random_bytes();
            group.bench_with_input(BenchmarkId::from_parameter(format!("{}b", $size * 8)), &$size, |bencher, _| {
                bencher.iter(|| black_box(hamming_array(black_box(&a), black_box(&b))))
            });
        }};
    }

    bench_size!(64);
    bench_size!(96);
    bench_size!(128);
    bench_size!(256);

    group.finish();
}

fn slice_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("input_type/dynamic_slice");

    for &size in &[64, 96, 128, 256] {
        let a = random_bytes_vec(size);
        let b = random_bytes_vec(size);
        group.bench_with_input(BenchmarkId::from_parameter(format!("{}b", size * 8)), &size, |bencher, _| {
            bencher.iter(|| black_box(hamming_slice(black_box(&a), black_box(&b))))
        });
    }

    group.finish();
}

criterion_group!(benches, array_benchmarks, slice_benchmarks);
criterion_main!(benches);

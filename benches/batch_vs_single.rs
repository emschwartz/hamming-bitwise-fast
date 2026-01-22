//! Compares batch operations vs individual function calls.
//!
//! Key insight: Batch APIs allow the compiler to optimize the entire loop,
//! potentially enabling better instruction scheduling and cache utilization.
//!
//! Run with: cargo bench --bench batch_vs_single
//! Quick mode: cargo bench --bench batch_vs_single -- --quick

mod helpers;

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use helpers::{random_bytes, random_bytes_array, random_bytes_vec};

const BATCH: usize = 64;

// ============================================================================
// Array implementations
// ============================================================================

/// Single array comparison using u64 chunks.
#[inline(always)]
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

/// Batch array comparison.
#[inline]
fn hamming_array_batch<const N: usize>(source: &[u8; N], targets: &[[u8; N]], out: &mut [u32]) {
    assert_eq!(targets.len(), out.len());
    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        *dist = hamming_array(source, target);
    }
}

// ============================================================================
// Slice implementations
// ============================================================================

/// Single slice comparison using u64 chunks.
#[inline(always)]
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

/// Batch slice comparison.
#[inline]
fn hamming_slice_batch(source: &[u8], targets: &[&[u8]], out: &mut [u32]) {
    assert_eq!(targets.len(), out.len());
    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        *dist = hamming_slice(source, target);
    }
}

// ============================================================================
// Array benchmarks
// ============================================================================

fn array_single_loop_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("array/single_loop");

    macro_rules! bench_size {
        ($size:expr) => {{
            let source: [u8; $size] = random_bytes();
            let targets: Vec<[u8; $size]> = random_bytes_array(BATCH);
            let mut out = vec![0u32; BATCH];
            group.bench_with_input(BenchmarkId::from_parameter($size), &$size, |bencher, _| {
                bencher.iter(|| {
                    for (target, dist) in black_box(&targets).iter().zip(out.iter_mut()) {
                        *dist = hamming_array(black_box(&source), target);
                    }
                    black_box(out[0])
                })
            });
        }};
    }

    bench_size!(64);
    bench_size!(96);
    bench_size!(128);
    bench_size!(256);

    group.finish();
}

fn array_batch_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("array/batch");

    macro_rules! bench_size {
        ($size:expr) => {{
            let source: [u8; $size] = random_bytes();
            let targets: Vec<[u8; $size]> = random_bytes_array(BATCH);
            let mut out = vec![0u32; BATCH];
            group.bench_with_input(BenchmarkId::from_parameter($size), &$size, |bencher, _| {
                bencher.iter(|| {
                    hamming_array_batch(black_box(&source), black_box(&targets), &mut out);
                    black_box(out[0])
                })
            });
        }};
    }

    bench_size!(64);
    bench_size!(96);
    bench_size!(128);
    bench_size!(256);

    group.finish();
}

// ============================================================================
// Slice benchmarks
// ============================================================================

fn slice_single_loop_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("slice/single_loop");

    for &size in &[64, 96, 128, 256] {
        let source = random_bytes_vec(size);
        let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(size)).collect();
        let mut out = vec![0u32; BATCH];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                for (target, dist) in black_box(&targets).iter().zip(out.iter_mut()) {
                    *dist = hamming_slice(black_box(&source), target);
                }
                black_box(out[0])
            })
        });
    }

    group.finish();
}

fn slice_batch_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("slice/batch");

    for &size in &[64, 96, 128, 256] {
        let source = random_bytes_vec(size);
        let targets: Vec<Vec<u8>> = (0..BATCH).map(|_| random_bytes_vec(size)).collect();
        let targets_refs: Vec<&[u8]> = targets.iter().map(|v| v.as_slice()).collect();
        let mut out = vec![0u32; BATCH];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                hamming_slice_batch(black_box(&source), black_box(&targets_refs), &mut out);
                black_box(out[0])
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    array_single_loop_benchmarks,
    array_batch_benchmarks,
    slice_single_loop_benchmarks,
    slice_batch_benchmarks
);
criterion_main!(benches);

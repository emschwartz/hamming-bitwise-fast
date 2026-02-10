//! Compares byte-by-byte iteration (u8) vs u64 chunk processing.
//!
//! Key insight: On ARM, simple u8 iteration auto-vectorizes well with NEON.
//! On x86, u64 chunk processing enables AVX-512 VPOPCNTDQ when available.
//!
//! Run with: cargo bench --bench chunk_strategy
//! Quick mode: cargo bench --bench chunk_strategy -- --quick
//! Compare with: RUSTFLAGS="-C target-cpu=native" cargo bench --bench chunk_strategy

mod helpers;

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use helpers::random_bytes;

// ============================================================================
// Implementations
// ============================================================================

/// Byte-by-byte iteration - simple, auto-vectorizes well on ARM.
#[inline]
fn hamming_u8_iter<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Process as u64 chunks - enables AVX-512 VPOPCNTDQ on x86.
#[inline]
fn hamming_u64_chunks<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);

    let main: u32 = a_chunks
        .clone()
        .zip(b_chunks.clone())
        .map(|(a_chunk, b_chunk)| {
            let a_val = u64::from_ne_bytes(a_chunk.try_into().unwrap());
            let b_val = u64::from_ne_bytes(b_chunk.try_into().unwrap());
            (a_val ^ b_val).count_ones()
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

fn benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_strategy");

    macro_rules! bench_size {
        ($size:expr) => {{
            let a: [u8; $size] = random_bytes();
            let b: [u8; $size] = random_bytes();
            let bits = format!("{}b", $size * 8);

            group.bench_with_input(BenchmarkId::new("byte_by_byte", &bits), &$size, |bencher, _| {
                bencher.iter(|| black_box(hamming_u8_iter(black_box(&a), black_box(&b))))
            });

            group.bench_with_input(BenchmarkId::new("u64_chunks", &bits), &$size, |bencher, _| {
                bencher.iter(|| black_box(hamming_u64_chunks(black_box(&a), black_box(&b))))
            });
        }};
    }

    bench_size!(64);
    bench_size!(96);
    bench_size!(128);
    bench_size!(256);

    group.finish();
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);

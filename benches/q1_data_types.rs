//! Q1: Are u64s or u8s better? Does it depend on the platform?
//!
//! Key findings to expect:
//! - ARM (aarch64): u8 arrays are faster due to NEON byte-level optimizations
//! - x86: u64 arrays are faster due to POPCNT operating on 64-bit values
//!
//! Run with: cargo bench --bench q1_data_types

mod helpers;

use helpers::*;

fn main() {
    divan::main();
}

// ============================================================================
// u8 array benchmarks
// Sizes: 64=512bit, 96=768bit, 128=1024bit, 256=2048bit (in bytes)
// ============================================================================

#[divan::bench(consts = [64, 96, 128, 256])]
fn u8_array<const N: usize>(bencher: divan::Bencher) {
    let a: [u8; N] = random_bytes();
    let b: [u8; N] = random_bytes();

    bencher.bench_local(|| {
        hamming_u8_iter(divan::black_box(&a), divan::black_box(&b))
    });
}

// ============================================================================
// u8 array with chunks_exact(8) - safe way to hint u64 processing
// ============================================================================

#[divan::bench(consts = [64, 96, 128, 256])]
fn u8_chunks_exact<const N: usize>(bencher: divan::Bencher) {
    let a: [u8; N] = random_bytes();
    let b: [u8; N] = random_bytes();

    bencher.bench_local(|| {
        hamming_u8_chunks(divan::black_box(&a), divan::black_box(&b))
    });
}

// ============================================================================
// u8 array with unsafe cast to u64 on x86
// Uses compile-time assertion to enforce N is a multiple of 8
// ============================================================================

#[divan::bench(consts = [64, 96, 128, 256])]
fn u8_as_u64_on_x86<const N: usize>(bencher: divan::Bencher) {
    let a: [u8; N] = random_bytes();
    let b: [u8; N] = random_bytes();

    bencher.bench_local(|| {
        hamming_u8_as_u64(divan::black_box(&a), divan::black_box(&b))
    });
}

// ============================================================================
// u64 array benchmarks (equivalent sizes)
// ============================================================================

#[divan::bench(consts = [8, 12, 16, 32])]
fn u64_array<const N: usize>(bencher: divan::Bencher) {
    let a: Embedding<N> = random_embedding();
    let b: Embedding<N> = random_embedding();

    bencher.bench_local(|| {
        hamming_u64_iter(divan::black_box(&a), divan::black_box(&b))
    });
}

// ============================================================================
// u64 array with unsafe cast to u8 on ARM
// This is the "best of both worlds" strategy: convenient u64 API, but
// uses u8 processing on ARM where NEON prefers byte operations.
// ============================================================================

#[divan::bench(consts = [8, 12, 16, 32])]
fn u64_as_u8_on_arm<const N: usize>(bencher: divan::Bencher) {
    let a: Embedding<N> = random_embedding();
    let b: Embedding<N> = random_embedding();

    bencher.bench_local(|| {
        hamming_u64_as_u8(divan::black_box(&a), divan::black_box(&b))
    });
}

// ============================================================================
// Library's hamming<N> function (uses platform-optimal strategy internally)
// ============================================================================

#[divan::bench(consts = [8, 12, 16, 32])]
fn library_hamming<const N: usize>(bencher: divan::Bencher) {
    let a: Embedding<N> = random_embedding();
    let b: Embedding<N> = random_embedding();

    bencher.bench_local(|| {
        hamming_bitwise_fast::hamming(divan::black_box(&a), divan::black_box(&b))
    });
}

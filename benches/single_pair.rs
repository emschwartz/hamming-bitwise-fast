mod implementations;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hamming_bitwise_fast::hamming_bitwise_fast;
use implementations::*;

// ============================================================================
// Group 1: Representation Benchmarks
// Key question: u8 slices vs u8 arrays vs u64 arrays?
// ============================================================================

fn bench_representation(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_pair/representation");
    group.throughput(Throughput::Elements(1));

    // Test at multiple embedding sizes to see if the answer changes
    bench_representation_size::<64, 8>(&mut group, "512bit");
    bench_representation_size::<96, 12>(&mut group, "768bit");
    bench_representation_size::<128, 16>(&mut group, "1024bit");
    bench_representation_size::<256, 32>(&mut group, "2048bit");

    group.finish();
}

fn bench_representation_size<const N_BYTES: usize, const N_U64S: usize>(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    size_name: &str,
) {
    // Generate test data
    let a_bytes: [u8; N_BYTES] = random_bytes();
    let b_bytes: [u8; N_BYTES] = random_bytes();
    let a_bytes_vec = a_bytes.to_vec();
    let b_bytes_vec = b_bytes.to_vec();
    let a_u64: Embedding<N_U64S> = bytes_to_embedding(&a_bytes);
    let b_u64: Embedding<N_U64S> = bytes_to_embedding(&b_bytes);

    // --- slice_baseline: Original hamming_bitwise_fast with byte slices ---
    // This is what most users would call first
    group.bench_with_input(
        BenchmarkId::new("slice_baseline", size_name),
        &(&a_bytes_vec, &b_bytes_vec),
        |bench, (a, b)| {
            bench.iter(|| {
                hamming_bitwise_fast(
                    criterion::black_box(a.as_slice()),
                    criterion::black_box(b.as_slice()),
                )
            })
        },
    );

    // --- u8_array: Fixed-size u8 array with iterator ---
    group.bench_with_input(
        BenchmarkId::new("u8_array", size_name),
        &(&a_bytes, &b_bytes),
        |bench, (a, b)| {
            bench.iter(|| hamming_u8_iter(criterion::black_box(a), criterion::black_box(b)))
        },
    );

    // --- u64_array: Fixed-size u64 array (best for auto-vectorization) ---
    group.bench_with_input(
        BenchmarkId::new("u64_array", size_name),
        &(&a_u64, &b_u64),
        |bench, (a, b)| {
            bench.iter(|| hamming_ref_iter(criterion::black_box(a), criterion::black_box(b)))
        },
    );
}

// ============================================================================
// Group 2: Dispatch Strategy Benchmarks
// Key question: Does multiversion/pulp help on this architecture?
// ============================================================================

fn bench_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_pair/dispatch");
    group.throughput(Throughput::Elements(1));

    // Test at multiple sizes - dispatch overhead may matter more for smaller inputs
    bench_dispatch_size::<8>(&mut group, "512bit");
    bench_dispatch_size::<12>(&mut group, "768bit");
    bench_dispatch_size::<16>(&mut group, "1024bit");
    bench_dispatch_size::<32>(&mut group, "2048bit");

    group.finish();
}

fn bench_dispatch_size<const N: usize>(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    size_name: &str,
) {
    let a: Embedding<N> = random_embedding();
    let b: Embedding<N> = random_embedding();

    // --- auto_vectorized: Default compilation, relies on compiler ---
    // Test this with different RUSTFLAGS to see compiler flag impact:
    //   Default: cargo bench --bench single_pair -- dispatch
    //   Native:  RUSTFLAGS="-C target-cpu=native" cargo bench --bench single_pair -- dispatch
    group.bench_with_input(
        BenchmarkId::new("auto_vectorized", size_name),
        &(&a, &b),
        |bench, (a, b)| {
            bench.iter(|| hamming_ref_iter(criterion::black_box(a), criterion::black_box(b)))
        },
    );

    // --- multiversion: Runtime CPU feature detection (mainly benefits x86) ---
    #[cfg(feature = "multiversion")]
    group.bench_with_input(
        BenchmarkId::new("multiversion", size_name),
        &(&a, &b),
        |bench, (a, b)| {
            bench.iter(|| hamming_multiversion(criterion::black_box(a), criterion::black_box(b)))
        },
    );

    // --- pulp: Portable SIMD abstraction ---
    group.bench_with_input(
        BenchmarkId::new("pulp", size_name),
        &(&a, &b),
        |bench, (a, b)| {
            bench.iter(|| hamming_pulp(criterion::black_box(a), criterion::black_box(b)))
        },
    );
}

criterion_group!(benches, bench_representation, bench_dispatch);
criterion_main!(benches);

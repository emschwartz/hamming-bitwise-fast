mod implementations;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hamming_bitwise_fast::hamming_bitwise_fast;
use implementations::*;

// ============================================================================
// Head-to-Head Comparison: Our implementations vs external crates
// Key question: What's the fastest Hamming distance for my use case?
// ============================================================================

fn bench_head_to_head(c: &mut Criterion) {
    let mut group = c.benchmark_group("head_to_head");
    group.throughput(Throughput::Elements(1));

    // Test at multiple embedding sizes
    bench_size::<64, 8>(&mut group, "512bit");
    bench_size::<96, 12>(&mut group, "768bit");
    bench_size::<128, 16>(&mut group, "1024bit");
    bench_size::<256, 32>(&mut group, "2048bit");

    group.finish();
}

fn bench_size<const N_BYTES: usize, const N_U64S: usize>(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    size_name: &str,
) {
    // Generate test data in both formats
    let a_bytes = random_bytes_vec(N_BYTES);
    let b_bytes = random_bytes_vec(N_BYTES);
    let a_u64: Embedding<N_U64S> = bytes_to_embedding(&a_bytes);
    let b_u64: Embedding<N_U64S> = bytes_to_embedding(&b_bytes);

    // --- hamming_bitwise_fast: Original slice-based (what users call first) ---
    group.bench_with_input(
        BenchmarkId::new("hamming_bitwise_fast", size_name),
        &(&a_bytes, &b_bytes),
        |bench, (a, b)| {
            bench.iter(|| {
                hamming_bitwise_fast(
                    criterion::black_box(a.as_slice()),
                    criterion::black_box(b.as_slice()),
                )
            })
        },
    );

    // --- our_best: Optimized u64 array version ---
    group.bench_with_input(
        BenchmarkId::new("our_best", size_name),
        &(&a_u64, &b_u64),
        |bench, (a, b)| {
            bench.iter(|| hamming_ref_iter(criterion::black_box(a), criterion::black_box(b)))
        },
    );

    // --- simsimd: External crate using SIMD intrinsics ---
    group.bench_with_input(
        BenchmarkId::new("simsimd", size_name),
        &(&a_bytes, &b_bytes),
        |bench, (a, b)| {
            bench.iter(|| {
                simsimd::BinarySimilarity::hamming(
                    criterion::black_box(a.as_slice()),
                    criterion::black_box(b.as_slice()),
                )
            })
        },
    );

    // --- hamming_crate: External crate (pure Rust) ---
    group.bench_with_input(
        BenchmarkId::new("hamming_crate", size_name),
        &(&a_bytes, &b_bytes),
        |bench, (a, b)| {
            bench.iter(|| {
                hamming::distance_fast(
                    criterion::black_box(a.as_slice()),
                    criterion::black_box(b.as_slice()),
                )
            })
        },
    );

    // --- triple_accel: External crate (SIMD-accelerated) ---
    group.bench_with_input(
        BenchmarkId::new("triple_accel", size_name),
        &(&a_bytes, &b_bytes),
        |bench, (a, b)| {
            bench.iter(|| {
                triple_accel::hamming(
                    criterion::black_box(a.as_slice()),
                    criterion::black_box(b.as_slice()),
                )
            })
        },
    );

    // --- hamming_rs: External crate (x86/x86_64 only, uses AVX2/AVX512) ---
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    group.bench_with_input(
        BenchmarkId::new("hamming_rs", size_name),
        &(&a_bytes, &b_bytes),
        |bench, (a, b)| {
            bench.iter(|| {
                hamming_rs::distance_faster(
                    criterion::black_box(a.as_slice()),
                    criterion::black_box(b.as_slice()),
                )
            })
        },
    );
}

criterion_group!(benches, bench_head_to_head);
criterion_main!(benches);

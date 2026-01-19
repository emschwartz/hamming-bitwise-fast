use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use hamming_bitwise_fast::*;
use rand::Rng;

// ============================================================================
// Test data generation
// ============================================================================

fn random_embedding<const N: usize>() -> Embedding<N> {
    let mut rng = rand::thread_rng();
    let mut emb = [0u64; N];
    for i in 0..N {
        emb[i] = rng.gen();
    }
    emb
}

fn random_bytes(size: usize) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen()).collect()
}

fn random_embeddings<const N: usize>(count: usize) -> Vec<Embedding<N>> {
    (0..count).map(|_| random_embedding()).collect()
}

// ============================================================================
// Combined Benchmark: Cumulative Optimization Impact
//
// Shows the progression from baseline to fully optimized:
// v1: Original slice-based implementation
// v2: Fixed-size [u64; N] type
// v3: + Multiversion dispatch (single pair)
// v4: + Batch function (amortized dispatch)
// v5: + Pre-allocated buffers
// ============================================================================

fn bench_cumulative_single_pair(c: &mut Criterion) {
    let mut group = c.benchmark_group("combined/single_pair_1024bit");

    // 1024-bit embeddings
    let a_bytes = random_bytes(128);
    let b_bytes = random_bytes(128);
    let a: Embedding<16> = bytes_to_embedding(&a_bytes);
    let b: Embedding<16> = bytes_to_embedding(&b_bytes);

    group.throughput(Throughput::Elements(1));

    // v1: Original slice-based (baseline)
    group.bench_function("v1_slice_baseline", |bench| {
        bench.iter(|| {
            hamming_bitwise_fast(
                criterion::black_box(&a_bytes),
                criterion::black_box(&b_bytes),
            )
        })
    });

    // v2: Fixed-size [u64; N] type (best auto-vectorized variant)
    group.bench_function("v2_fixed_type", |bench| {
        bench.iter(|| {
            hamming_ref_iter(
                criterion::black_box(&a),
                criterion::black_box(&b),
            )
        })
    });

    // v3: + Multiversion dispatch
    #[cfg(feature = "multiversion")]
    group.bench_function("v3_multiversion", |bench| {
        bench.iter(|| {
            hamming_multiversion(
                criterion::black_box(&a),
                criterion::black_box(&b),
            )
        })
    });

    group.finish();
}

fn bench_cumulative_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("combined/batch_1000_embeddings");

    const N: usize = 16;
    let source: Embedding<N> = random_embedding();
    let source_bytes: Vec<u8> = source.iter()
        .flat_map(|x| x.to_ne_bytes())
        .collect();
    let targets: Vec<Embedding<N>> = random_embeddings(1000);
    let targets_bytes: Vec<Vec<u8>> = targets.iter()
        .map(|t| t.iter().flat_map(|x| x.to_ne_bytes()).collect())
        .collect();

    group.throughput(Throughput::Elements(1000));

    // v1: Loop with slice-based hamming (baseline)
    group.bench_function("v1_loop_slice", |bench| {
        let mut out = vec![0u32; 1000];
        bench.iter(|| {
            for (i, target) in targets_bytes.iter().enumerate() {
                out[i] = hamming_bitwise_fast(
                    criterion::black_box(&source_bytes),
                    criterion::black_box(target),
                );
            }
            out[0]
        })
    });

    // v2: Loop with fixed-size type
    group.bench_function("v2_loop_fixed_type", |bench| {
        let mut out = vec![0u32; 1000];
        bench.iter(|| {
            for (i, target) in targets.iter().enumerate() {
                out[i] = hamming_ref_iter(
                    criterion::black_box(&source),
                    criterion::black_box(target),
                );
            }
            out[0]
        })
    });

    // v3: Loop with multiversion (N dispatches)
    #[cfg(feature = "multiversion")]
    group.bench_function("v3_loop_multiversion", |bench| {
        let mut out = vec![0u32; 1000];
        bench.iter(|| {
            for (i, target) in targets.iter().enumerate() {
                out[i] = hamming_multiversion(
                    criterion::black_box(&source),
                    criterion::black_box(target),
                );
            }
            out[0]
        })
    });

    // v4: Batch function (1 dispatch) - still allocating per call
    group.bench_function("v4_batch_alloc_per_call", |bench| {
        bench.iter(|| {
            let mut out = vec![0u32; 1000];
            hamming_batch_into_auto(
                criterion::black_box(&source),
                criterion::black_box(&targets),
                &mut out,
            );
            out[0]
        })
    });

    // v4 with multiversion
    #[cfg(feature = "multiversion")]
    group.bench_function("v4_batch_multiversion_alloc", |bench| {
        bench.iter(|| {
            let mut out = vec![0u32; 1000];
            hamming_batch_into(
                criterion::black_box(&source),
                criterion::black_box(&targets),
                &mut out,
            );
            out[0]
        })
    });

    // v5: Batch + pre-allocated buffer (full optimization)
    group.bench_function("v5_batch_preallocated", |bench| {
        let mut out = vec![0u32; 1000];
        bench.iter(|| {
            hamming_batch_into_auto(
                criterion::black_box(&source),
                criterion::black_box(&targets),
                &mut out,
            );
            out[0]
        })
    });

    // v5 with multiversion (the best)
    #[cfg(feature = "multiversion")]
    group.bench_function("v5_full_optimized", |bench| {
        let mut out = vec![0u32; 1000];
        bench.iter(|| {
            hamming_batch_into(
                criterion::black_box(&source),
                criterion::black_box(&targets),
                &mut out,
            );
            out[0]
        })
    });

    group.finish();
}

// ============================================================================
// Summary: Single table showing all optimization levels
// ============================================================================

fn bench_optimization_summary(c: &mut Criterion) {
    let mut group = c.benchmark_group("combined/optimization_summary");

    const N: usize = 16;
    let source: Embedding<N> = random_embedding();
    let targets: Vec<Embedding<N>> = random_embeddings(1000);

    // Pre-allocate for the optimized versions
    let mut preallocated_out = vec![0u32; 1000];

    group.throughput(Throughput::Elements(1000));

    // Baseline: what you'd write naively
    group.bench_function("baseline_naive", |bench| {
        let source_bytes: Vec<u8> = source.iter().flat_map(|x| x.to_ne_bytes()).collect();
        let targets_bytes: Vec<Vec<u8>> = targets.iter()
            .map(|t| t.iter().flat_map(|x| x.to_ne_bytes()).collect())
            .collect();
        bench.iter(|| {
            let distances: Vec<u32> = targets_bytes.iter()
                .map(|t| hamming_bitwise_fast(&source_bytes, t))
                .collect();
            distances[0]
        })
    });

    // Best: all optimizations combined
    #[cfg(feature = "multiversion")]
    group.bench_function("best_all_optimizations", |bench| {
        bench.iter(|| {
            hamming_batch_into(
                criterion::black_box(&source),
                criterion::black_box(&targets),
                &mut preallocated_out,
            );
            preallocated_out[0]
        })
    });

    // Best without multiversion (for comparison)
    group.bench_function("best_without_multiversion", |bench| {
        bench.iter(|| {
            hamming_batch_into_auto(
                criterion::black_box(&source),
                criterion::black_box(&targets),
                &mut preallocated_out,
            );
            preallocated_out[0]
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_cumulative_single_pair,
    bench_cumulative_batch,
    bench_optimization_summary
);
criterion_main!(benches);

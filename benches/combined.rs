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
    let mut group = c.benchmark_group("combined/batch_per_comparison");

    const N: usize = 16;
    const BATCH_SIZE: usize = 1000;
    let source: Embedding<N> = random_embedding();
    let source_bytes: Vec<u8> = source.iter()
        .flat_map(|x| x.to_ne_bytes())
        .collect();
    let targets: Vec<Embedding<N>> = random_embeddings(BATCH_SIZE);
    let targets_bytes: Vec<Vec<u8>> = targets.iter()
        .map(|t| t.iter().flat_map(|x| x.to_ne_bytes()).collect())
        .collect();

    // Use Elements(1) - each iteration processes 1 comparison (we'll iterate BATCH_SIZE times)
    group.throughput(Throughput::Elements(1));

    // v1: Loop with slice-based hamming (baseline) - per comparison
    group.bench_function("v1_loop_slice", |bench| {
        let mut idx = 0usize;
        bench.iter(|| {
            let result = hamming_bitwise_fast(
                criterion::black_box(&source_bytes),
                criterion::black_box(&targets_bytes[idx]),
            );
            idx = (idx + 1) % BATCH_SIZE;
            result
        })
    });

    // v2: Loop with fixed-size type - per comparison
    group.bench_function("v2_loop_fixed_type", |bench| {
        let mut idx = 0usize;
        bench.iter(|| {
            let result = hamming_ref_iter(
                criterion::black_box(&source),
                criterion::black_box(&targets[idx]),
            );
            idx = (idx + 1) % BATCH_SIZE;
            result
        })
    });

    // v3: Loop with multiversion - per comparison
    #[cfg(feature = "multiversion")]
    group.bench_function("v3_loop_multiversion", |bench| {
        let mut idx = 0usize;
        bench.iter(|| {
            let result = hamming_multiversion(
                criterion::black_box(&source),
                criterion::black_box(&targets[idx]),
            );
            idx = (idx + 1) % BATCH_SIZE;
            result
        })
    });

    // v4: Batch function - amortized per comparison using iter_custom
    group.bench_function("v4_batch_alloc_per_call", |bench| {
        bench.iter_custom(|iters| {
            let iters = iters as usize;
            let batches = (iters + BATCH_SIZE - 1) / BATCH_SIZE;
            let actual_comparisons = batches * BATCH_SIZE;
            let start = std::time::Instant::now();
            for _ in 0..batches {
                let mut out = vec![0u32; BATCH_SIZE];
                hamming_batch_into_auto(
                    criterion::black_box(&source),
                    criterion::black_box(&targets),
                    &mut out,
                );
                criterion::black_box(&out);
            }
            // Scale time to match requested iters
            start.elapsed().mul_f64(iters as f64 / actual_comparisons as f64)
        })
    });

    // v4 with multiversion - amortized per comparison
    #[cfg(feature = "multiversion")]
    group.bench_function("v4_batch_multiversion_alloc", |bench| {
        bench.iter_custom(|iters| {
            let iters = iters as usize;
            let batches = (iters + BATCH_SIZE - 1) / BATCH_SIZE;
            let actual_comparisons = batches * BATCH_SIZE;
            let start = std::time::Instant::now();
            for _ in 0..batches {
                let mut out = vec![0u32; BATCH_SIZE];
                hamming_batch_into(
                    criterion::black_box(&source),
                    criterion::black_box(&targets),
                    &mut out,
                );
                criterion::black_box(&out);
            }
            start.elapsed().mul_f64(iters as f64 / actual_comparisons as f64)
        })
    });

    // v5: Batch + pre-allocated buffer - amortized per comparison
    group.bench_function("v5_batch_preallocated", |bench| {
        let mut out = vec![0u32; BATCH_SIZE];
        bench.iter_custom(|iters| {
            let iters = iters as usize;
            let batches = (iters + BATCH_SIZE - 1) / BATCH_SIZE;
            let actual_comparisons = batches * BATCH_SIZE;
            let start = std::time::Instant::now();
            for _ in 0..batches {
                hamming_batch_into_auto(
                    criterion::black_box(&source),
                    criterion::black_box(&targets),
                    &mut out,
                );
                criterion::black_box(&out);
            }
            start.elapsed().mul_f64(iters as f64 / actual_comparisons as f64)
        })
    });

    // v5 with multiversion (the best) - amortized per comparison
    #[cfg(feature = "multiversion")]
    group.bench_function("v5_full_optimized", |bench| {
        let mut out = vec![0u32; BATCH_SIZE];
        bench.iter_custom(|iters| {
            let iters = iters as usize;
            let batches = (iters + BATCH_SIZE - 1) / BATCH_SIZE;
            let actual_comparisons = batches * BATCH_SIZE;
            let start = std::time::Instant::now();
            for _ in 0..batches {
                hamming_batch_into(
                    criterion::black_box(&source),
                    criterion::black_box(&targets),
                    &mut out,
                );
                criterion::black_box(&out);
            }
            start.elapsed().mul_f64(iters as f64 / actual_comparisons as f64)
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
    const BATCH_SIZE: usize = 1000;
    let source: Embedding<N> = random_embedding();
    let targets: Vec<Embedding<N>> = random_embeddings(BATCH_SIZE);

    // Pre-allocate for the optimized versions
    let mut preallocated_out = vec![0u32; BATCH_SIZE];

    // Per-comparison throughput
    group.throughput(Throughput::Elements(1));

    // Baseline: what you'd write naively - per comparison
    group.bench_function("baseline_naive", |bench| {
        let source_bytes: Vec<u8> = source.iter().flat_map(|x| x.to_ne_bytes()).collect();
        let targets_bytes: Vec<Vec<u8>> = targets.iter()
            .map(|t| t.iter().flat_map(|x| x.to_ne_bytes()).collect())
            .collect();
        let mut idx = 0usize;
        bench.iter(|| {
            let result = hamming_bitwise_fast(&source_bytes, &targets_bytes[idx]);
            idx = (idx + 1) % BATCH_SIZE;
            result
        })
    });

    // Best: all optimizations combined - amortized per comparison
    #[cfg(feature = "multiversion")]
    group.bench_function("best_all_optimizations", |bench| {
        bench.iter_custom(|iters| {
            let iters = iters as usize;
            let batches = (iters + BATCH_SIZE - 1) / BATCH_SIZE;
            let actual_comparisons = batches * BATCH_SIZE;
            let start = std::time::Instant::now();
            for _ in 0..batches {
                hamming_batch_into(
                    criterion::black_box(&source),
                    criterion::black_box(&targets),
                    &mut preallocated_out,
                );
                criterion::black_box(&preallocated_out);
            }
            start.elapsed().mul_f64(iters as f64 / actual_comparisons as f64)
        })
    });

    // Best without multiversion (for comparison) - amortized per comparison
    group.bench_function("best_without_multiversion", |bench| {
        bench.iter_custom(|iters| {
            let iters = iters as usize;
            let batches = (iters + BATCH_SIZE - 1) / BATCH_SIZE;
            let actual_comparisons = batches * BATCH_SIZE;
            let start = std::time::Instant::now();
            for _ in 0..batches {
                hamming_batch_into_auto(
                    criterion::black_box(&source),
                    criterion::black_box(&targets),
                    &mut preallocated_out,
                );
                criterion::black_box(&preallocated_out);
            }
            start.elapsed().mul_f64(iters as f64 / actual_comparisons as f64)
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

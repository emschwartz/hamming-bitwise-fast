mod implementations;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use hamming_bitwise_fast::hamming_bitwise_fast;
use implementations::*;

// ============================================================================
// Optimization Summary: Quick before/after comparison
// Shows the full impact of all optimizations combined
// ============================================================================

fn bench_optimization_summary(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_summary");

    const N: usize = 16; // 1024-bit embeddings
    const BATCH_SIZE: usize = 1000;
    let source: Embedding<N> = random_embedding();
    let targets: Vec<Embedding<N>> = random_embeddings(BATCH_SIZE);

    // Convert to bytes for baseline comparison
    let source_bytes: Vec<u8> = source.iter().flat_map(|x| x.to_ne_bytes()).collect();
    let targets_bytes: Vec<Vec<u8>> = targets
        .iter()
        .map(|t| t.iter().flat_map(|x| x.to_ne_bytes()).collect())
        .collect();

    // Per-comparison throughput for direct comparison
    group.throughput(Throughput::Elements(1));

    // --- baseline_naive: What most users would write first ---
    // Loop calling slice-based hamming_bitwise_fast
    group.bench_function("baseline_naive", |bench| {
        let mut idx = 0usize;
        bench.iter(|| {
            let result = hamming_bitwise_fast(&source_bytes, &targets_bytes[idx]);
            idx = (idx + 1) % BATCH_SIZE;
            result
        })
    });

    // --- best_optimized: All optimizations applied ---
    // u64 array + batch function + preallocated buffer
    group.bench_function("best_optimized", |bench| {
        let mut out = vec![0u32; BATCH_SIZE];
        bench.iter_custom(|iters| {
            let iters = iters as usize;
            let batches = (iters + BATCH_SIZE - 1) / BATCH_SIZE;
            let actual = batches * BATCH_SIZE;
            let start = std::time::Instant::now();
            for _ in 0..batches {
                hamming_batch_into_auto(
                    criterion::black_box(&source),
                    criterion::black_box(&targets),
                    &mut out,
                );
                criterion::black_box(&out);
            }
            start.elapsed().mul_f64(iters as f64 / actual as f64)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_optimization_summary);
criterion_main!(benches);

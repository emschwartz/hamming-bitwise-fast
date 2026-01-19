mod implementations;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use implementations::*;

// ============================================================================
// Group 1: Batch Operations
// Key question: How much faster is hamming_batch_into vs looping single calls?
// ============================================================================

fn bench_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch/1000");

    const N: usize = 16; // 1024-bit embeddings
    const BATCH_SIZE: usize = 1000;
    let source: Embedding<N> = random_embedding();
    let targets = random_embeddings::<N>(BATCH_SIZE);

    // Per-comparison throughput to make numbers directly comparable
    group.throughput(Throughput::Elements(1));

    // --- loop_single: Naive approach - loop calling single hamming ---
    // This is what most users would write first
    group.bench_function("loop_single", |bench| {
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

    // --- batch_function: Recommended approach - single dispatch for all ---
    // Amortizes function call overhead across batch
    group.bench_function("batch_function", |bench| {
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

// ============================================================================
// Group 2: Allocation Patterns
// Key question: How much does allocation strategy matter?
// ============================================================================

fn bench_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_allocation/1000");

    const N: usize = 16;
    const BATCH_SIZE: usize = 1000;
    let source: Embedding<N> = random_embedding();
    let targets = random_embeddings::<N>(BATCH_SIZE);

    // Per-comparison throughput
    group.throughput(Throughput::Elements(1));

    // --- alloc_per_call: Anti-pattern - allocate output buffer each time ---
    // Shows cost of repeated allocation
    group.bench_function("alloc_per_call", |bench| {
        bench.iter_custom(|iters| {
            let iters = iters as usize;
            let batches = (iters + BATCH_SIZE - 1) / BATCH_SIZE;
            let actual = batches * BATCH_SIZE;
            let start = std::time::Instant::now();
            for _ in 0..batches {
                let mut out = vec![0u32; BATCH_SIZE]; // Allocation inside loop
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

    // --- preallocated: Best practice - reuse output buffer ---
    group.bench_function("preallocated", |bench| {
        let mut out = vec![0u32; BATCH_SIZE]; // Allocation outside loop
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

criterion_group!(benches, bench_batch, bench_allocation);
criterion_main!(benches);

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
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

fn random_embeddings<const N: usize>(count: usize) -> Vec<Embedding<N>> {
    (0..count).map(|_| random_embedding()).collect()
}

// ============================================================================
// Group 2A: Batch Dispatch Overhead
// Compares: loop of single calls vs batch function with single dispatch
// ============================================================================

fn bench_batch_dispatch_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch/dispatch_overhead_per_comparison");

    const N: usize = 16; // 1024-bit embeddings
    let source: Embedding<N> = random_embedding();

    // Per-comparison throughput
    group.throughput(Throughput::Elements(1));

    for batch_size in [100, 500, 1000, 5000] {
        let targets = random_embeddings::<N>(batch_size);

        // Loop calling single hamming - per comparison
        group.bench_with_input(
            BenchmarkId::new("loop_single_ref_iter", batch_size),
            &(&source, &targets, batch_size),
            |bench, (source, targets, bs)| {
                let mut idx = 0usize;
                bench.iter(|| {
                    let result = hamming_ref_iter(
                        criterion::black_box(source),
                        criterion::black_box(&targets[idx]),
                    );
                    idx = (idx + 1) % *bs;
                    result
                })
            },
        );

        // Loop calling multiversion single - per comparison
        #[cfg(feature = "multiversion")]
        group.bench_with_input(
            BenchmarkId::new("loop_single_multiversion", batch_size),
            &(&source, &targets, batch_size),
            |bench, (source, targets, bs)| {
                let mut idx = 0usize;
                bench.iter(|| {
                    let result = hamming_multiversion(
                        criterion::black_box(source),
                        criterion::black_box(&targets[idx]),
                    );
                    idx = (idx + 1) % *bs;
                    result
                })
            },
        );

        // Batch function with auto-vectorization - amortized per comparison
        group.bench_with_input(
            BenchmarkId::new("batch_auto", batch_size),
            &(&source, &targets, batch_size),
            |bench, (source, targets, bs)| {
                let mut out = vec![0u32; *bs];
                bench.iter_custom(|iters| {
                    let iters = iters as usize;
                    let batches = (iters + *bs - 1) / *bs;
                    let actual = batches * *bs;
                    let start = std::time::Instant::now();
                    for _ in 0..batches {
                        hamming_batch_into_auto(
                            criterion::black_box(source),
                            criterion::black_box(targets),
                            &mut out,
                        );
                        criterion::black_box(&out);
                    }
                    start.elapsed().mul_f64(iters as f64 / actual as f64)
                })
            },
        );

        // Batch function with multiversion - amortized per comparison
        #[cfg(feature = "multiversion")]
        group.bench_with_input(
            BenchmarkId::new("batch_multiversion", batch_size),
            &(&source, &targets, batch_size),
            |bench, (source, targets, bs)| {
                let mut out = vec![0u32; *bs];
                bench.iter_custom(|iters| {
                    let iters = iters as usize;
                    let batches = (iters + *bs - 1) / *bs;
                    let actual = batches * *bs;
                    let start = std::time::Instant::now();
                    for _ in 0..batches {
                        hamming_batch_into(
                            criterion::black_box(source),
                            criterion::black_box(targets),
                            &mut out,
                        );
                        criterion::black_box(&out);
                    }
                    start.elapsed().mul_f64(iters as f64 / actual as f64)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Group 2B: Batch Size Exploration
// Tests hypothesis: fixed-size batches might enable better unrolling
// ============================================================================

fn bench_batch_size_exploration(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch/size_exploration_per_comparison");

    const N: usize = 16; // 1024-bit embeddings
    let source: Embedding<N> = random_embedding();

    // Per-comparison throughput
    group.throughput(Throughput::Elements(1));

    // Test different batch sizes to find optimal
    // L1 cache is ~32KB, 64 embeddings x 128 bytes = 8KB
    for batch_size in [16, 32, 64, 128, 256, 512] {
        let targets = random_embeddings::<N>(batch_size);

        // Dynamic batch (slice) - amortized per comparison
        group.bench_with_input(
            BenchmarkId::new("dynamic_batch", batch_size),
            &(&source, &targets, batch_size),
            |bench, (source, targets, bs)| {
                let mut out = vec![0u32; *bs];
                bench.iter_custom(|iters| {
                    let iters = iters as usize;
                    let batches = (iters + *bs - 1) / *bs;
                    let actual = batches * *bs;
                    let start = std::time::Instant::now();
                    for _ in 0..batches {
                        hamming_batch_into_auto(
                            criterion::black_box(source),
                            criterion::black_box(targets),
                            &mut out,
                        );
                        criterion::black_box(&out);
                    }
                    start.elapsed().mul_f64(iters as f64 / actual as f64)
                })
            },
        );
    }

    // Fixed-size batch comparisons (requires const generics, so we do specific sizes)
    bench_fixed_batch_64(&mut group, &source);
    bench_fixed_batch_128(&mut group, &source);

    group.finish();
}

fn bench_fixed_batch_64(group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>, source: &Embedding<16>) {
    const BATCH: usize = 64;
    let targets: [Embedding<16>; BATCH] = std::array::from_fn(|_| random_embedding());
    let mut out = [0u32; BATCH];

    group.bench_function("fixed_64_auto", |bench| {
        bench.iter_custom(|iters| {
            let iters = iters as usize;
            let batches = (iters + BATCH - 1) / BATCH;
            let actual = batches * BATCH;
            let start = std::time::Instant::now();
            for _ in 0..batches {
                hamming_batch_fixed_auto(
                    criterion::black_box(source),
                    criterion::black_box(&targets),
                    &mut out,
                );
                criterion::black_box(&out);
            }
            start.elapsed().mul_f64(iters as f64 / actual as f64)
        })
    });

    #[cfg(feature = "multiversion")]
    group.bench_function("fixed_64_multiversion", |bench| {
        bench.iter_custom(|iters| {
            let iters = iters as usize;
            let batches = (iters + BATCH - 1) / BATCH;
            let actual = batches * BATCH;
            let start = std::time::Instant::now();
            for _ in 0..batches {
                hamming_batch_fixed(
                    criterion::black_box(source),
                    criterion::black_box(&targets),
                    &mut out,
                );
                criterion::black_box(&out);
            }
            start.elapsed().mul_f64(iters as f64 / actual as f64)
        })
    });
}

fn bench_fixed_batch_128(group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>, source: &Embedding<16>) {
    const BATCH: usize = 128;
    let targets: [Embedding<16>; BATCH] = std::array::from_fn(|_| random_embedding());
    let mut out = [0u32; BATCH];

    group.bench_function("fixed_128_auto", |bench| {
        bench.iter_custom(|iters| {
            let iters = iters as usize;
            let batches = (iters + BATCH - 1) / BATCH;
            let actual = batches * BATCH;
            let start = std::time::Instant::now();
            for _ in 0..batches {
                hamming_batch_fixed_auto(
                    criterion::black_box(source),
                    criterion::black_box(&targets),
                    &mut out,
                );
                criterion::black_box(&out);
            }
            start.elapsed().mul_f64(iters as f64 / actual as f64)
        })
    });

    #[cfg(feature = "multiversion")]
    group.bench_function("fixed_128_multiversion", |bench| {
        bench.iter_custom(|iters| {
            let iters = iters as usize;
            let batches = (iters + BATCH - 1) / BATCH;
            let actual = batches * BATCH;
            let start = std::time::Instant::now();
            for _ in 0..batches {
                hamming_batch_fixed(
                    criterion::black_box(source),
                    criterion::black_box(&targets),
                    &mut out,
                );
                criterion::black_box(&out);
            }
            start.elapsed().mul_f64(iters as f64 / actual as f64)
        })
    });
}

criterion_group!(
    benches,
    bench_batch_dispatch_overhead,
    bench_batch_size_exploration
);
criterion_main!(benches);

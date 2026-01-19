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
// Group 3A: Cache Layout Benchmarks
// Compares memory layouts for batch operations
// ============================================================================

/// Interleaved layout: indices and embeddings together
struct InterleavedItem<const N: usize> {
    index: usize,
    embedding: Embedding<N>,
}

/// Separate layout: indices in one array, embeddings in another (cache-friendly)
struct SeparateLayout<const N: usize> {
    indices: Vec<usize>,
    embeddings: Vec<Embedding<N>>,
}

fn bench_cache_layout(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/cache_layout_per_comparison");

    const N: usize = 16; // 1024-bit embeddings
    const BATCH_SIZE: usize = 1000;
    let source: Embedding<N> = random_embedding();

    // Create interleaved layout
    let interleaved: Vec<InterleavedItem<N>> = (0..BATCH_SIZE)
        .map(|i| InterleavedItem {
            index: i,
            embedding: random_embedding(),
        })
        .collect();

    // Create separate layout (cache-friendly)
    let separate = SeparateLayout {
        indices: (0..BATCH_SIZE).collect(),
        embeddings: random_embeddings(BATCH_SIZE),
    };

    // Contiguous embeddings only (most cache-friendly for distance computation)
    let contiguous: Vec<Embedding<N>> = random_embeddings(BATCH_SIZE);

    // Per-comparison throughput
    group.throughput(Throughput::Elements(1));

    // Interleaved: Vec<(usize, Embedding)> - 136 bytes per element
    // Cache unfriendly: large stride between embeddings - per comparison
    group.bench_function("interleaved_136byte_stride", |bench| {
        let mut idx = 0usize;
        bench.iter(|| {
            let result = hamming_ref_iter(
                criterion::black_box(&source),
                criterion::black_box(&interleaved[idx].embedding),
            );
            idx = (idx + 1) % BATCH_SIZE;
            result
        })
    });

    // Separate indices + embeddings - per comparison
    group.bench_function("separate_arrays", |bench| {
        let mut idx = 0usize;
        bench.iter(|| {
            let result = hamming_ref_iter(
                criterion::black_box(&source),
                criterion::black_box(&separate.embeddings[idx]),
            );
            idx = (idx + 1) % BATCH_SIZE;
            result
        })
    });

    // Contiguous embeddings only - most cache-friendly - per comparison
    group.bench_function("contiguous_embeddings", |bench| {
        let mut idx = 0usize;
        bench.iter(|| {
            let result = hamming_ref_iter(
                criterion::black_box(&source),
                criterion::black_box(&contiguous[idx]),
            );
            idx = (idx + 1) % BATCH_SIZE;
            result
        })
    });

    // Contiguous with batch function - amortized per comparison
    group.bench_function("contiguous_batch_fn", |bench| {
        let mut out = vec![0u32; BATCH_SIZE];
        bench.iter_custom(|iters| {
            let iters = iters as usize;
            let batches = (iters + BATCH_SIZE - 1) / BATCH_SIZE;
            let actual = batches * BATCH_SIZE;
            let start = std::time::Instant::now();
            for _ in 0..batches {
                hamming_batch_into_auto(
                    criterion::black_box(&source),
                    criterion::black_box(&contiguous),
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
// Group 3B: Allocation Strategy Benchmarks
// Compares: per-call allocation vs pre-allocated vs stack-allocated
// ============================================================================

fn bench_allocation_strategy(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/allocation_strategy_per_comparison");

    const N: usize = 16;
    let source: Embedding<N> = random_embedding();

    // Per-comparison throughput
    group.throughput(Throughput::Elements(1));

    for batch_size in [100, 500, 1000] {
        let targets = random_embeddings::<N>(batch_size);

        // Per-call allocation: allocate Vec<u32> each time - amortized per comparison
        group.bench_with_input(
            BenchmarkId::new("alloc_per_call", batch_size),
            &(&source, &targets, batch_size),
            |bench, (source, targets, bs)| {
                bench.iter_custom(|iters| {
                    let iters = iters as usize;
                    let batches = (iters + *bs - 1) / *bs;
                    let actual = batches * *bs;
                    let start = std::time::Instant::now();
                    for _ in 0..batches {
                        let mut out = vec![0u32; *bs]; // Allocation inside loop
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

        // Pre-allocated: reuse Vec<u32> across calls - amortized per comparison
        group.bench_with_input(
            BenchmarkId::new("preallocated_vec", batch_size),
            &(&source, &targets, batch_size),
            |bench, (source, targets, bs)| {
                let mut out = vec![0u32; *bs]; // Allocation outside loop
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

    // Fixed-size stack allocation (only for small batches)
    bench_stack_allocation(&mut group, &source);

    group.finish();
}

fn bench_stack_allocation(group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>, source: &Embedding<16>) {
    const BATCH: usize = 64;
    let targets: [Embedding<16>; BATCH] = std::array::from_fn(|_| random_embedding());

    // Stack-allocated output array - amortized per comparison
    group.bench_function("stack_array_64", |bench| {
        bench.iter_custom(|iters| {
            let iters = iters as usize;
            let batches = (iters + BATCH - 1) / BATCH;
            let actual = batches * BATCH;
            let start = std::time::Instant::now();
            for _ in 0..batches {
                let mut out = [0u32; BATCH]; // On stack, no heap allocation
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

    // Compare with heap-allocated for same size (using boxed array) - amortized per comparison
    group.bench_function("heap_boxed_array_64", |bench| {
        bench.iter_custom(|iters| {
            let iters = iters as usize;
            let batches = (iters + BATCH - 1) / BATCH;
            let actual = batches * BATCH;
            let start = std::time::Instant::now();
            for _ in 0..batches {
                let mut out = Box::new([0u32; BATCH]); // Heap allocation of fixed-size array
                hamming_batch_fixed_auto(
                    criterion::black_box(source),
                    criterion::black_box(&targets),
                    &mut *out,
                );
                criterion::black_box(&out);
            }
            start.elapsed().mul_f64(iters as f64 / actual as f64)
        })
    });

    // Pre-allocated heap for comparison (boxed array) - amortized per comparison
    group.bench_function("preallocated_boxed_64", |bench| {
        let mut out = Box::new([0u32; BATCH]);
        bench.iter_custom(|iters| {
            let iters = iters as usize;
            let batches = (iters + BATCH - 1) / BATCH;
            let actual = batches * BATCH;
            let start = std::time::Instant::now();
            for _ in 0..batches {
                hamming_batch_fixed_auto(
                    criterion::black_box(source),
                    criterion::black_box(&targets),
                    &mut *out,
                );
                criterion::black_box(&out);
            }
            start.elapsed().mul_f64(iters as f64 / actual as f64)
        })
    });
}

// ============================================================================
// Bonus: Top-K with different allocation strategies
// ============================================================================

fn bench_topk_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/topk_allocation_per_comparison");

    const N: usize = 16;
    const K: usize = 10;
    const BATCH_SIZE: usize = 1000;
    let source: Embedding<N> = random_embedding();
    let targets = random_embeddings::<N>(BATCH_SIZE);

    // Per-comparison throughput
    group.throughput(Throughput::Elements(1));

    // Top-K with Vec allocation - amortized per comparison
    group.bench_function("topk_vec", |bench| {
        bench.iter_custom(|iters| {
            let iters = iters as usize;
            let batches = (iters + BATCH_SIZE - 1) / BATCH_SIZE;
            let actual = batches * BATCH_SIZE;
            let start = std::time::Instant::now();
            for _ in 0..batches {
                let mut distances: Vec<(u32, usize)> = targets
                    .iter()
                    .enumerate()
                    .map(|(i, t)| (hamming_ref_iter(&source, t), i))
                    .collect();
                distances.sort_by_key(|(d, _)| *d);
                let topk: Vec<(u32, usize)> = distances.into_iter().take(K).collect();
                criterion::black_box(topk[0].0);
            }
            start.elapsed().mul_f64(iters as f64 / actual as f64)
        })
    });

    // Top-K with fixed-size array (no heap allocation in hot path) - amortized per comparison
    group.bench_function("topk_fixed_array", |bench| {
        bench.iter_custom(|iters| {
            let iters = iters as usize;
            let batches = (iters + BATCH_SIZE - 1) / BATCH_SIZE;
            let actual = batches * BATCH_SIZE;
            let start = std::time::Instant::now();
            for _ in 0..batches {
                let mut topk = [(u32::MAX, usize::MAX); K];
                for (i, target) in targets.iter().enumerate() {
                    let dist = hamming_ref_iter(&source, target);
                    // Insert into sorted array if smaller than largest
                    if dist < topk[K - 1].0 {
                        topk[K - 1] = (dist, i);
                        // Bubble sort to maintain order (efficient for small K)
                        for j in (1..K).rev() {
                            if topk[j].0 < topk[j - 1].0 {
                                topk.swap(j, j - 1);
                            } else {
                                break;
                            }
                        }
                    }
                }
                criterion::black_box(topk[0].0);
            }
            start.elapsed().mul_f64(iters as f64 / actual as f64)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_cache_layout,
    bench_allocation_strategy,
    bench_topk_allocation
);
criterion_main!(benches);

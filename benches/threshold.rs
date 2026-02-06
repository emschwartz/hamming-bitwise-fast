//! Benchmarks for early-exit threshold APIs.
//!
//! Run with: cargo bench --bench threshold
//! Quick mode: cargo bench --bench threshold -- --quick

mod helpers;

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use hamming_bitwise_fast::array;
use helpers::{random_bytes, random_bytes_array};

const BATCH: usize = 1000;

// ============================================================================
// Early-exit threshold
// ============================================================================

fn early_exit_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("early_exit");

    // 1024-bit vectors
    {
        let source: [u8; 128] = random_bytes();
        let targets: Vec<[u8; 128]> = random_bytes_array(BATCH);

        // Baseline: full computation (no early exit)
        group.bench_function("full_compute_1024b", |bencher| {
            let mut out = vec![0u32; BATCH];
            bencher.iter(|| {
                for (target, dist) in black_box(&targets).iter().zip(out.iter_mut()) {
                    *dist = array::distance(black_box(&source), target);
                }
                black_box(out[0])
            })
        });

        // Tight threshold: ~10% of max (should reject most random vectors)
        // Random vectors average ~50% bits different (512 out of 1024),
        // so a threshold of 100 rejects nearly all.
        group.bench_function("within_thresh100_1024b", |bencher| {
            bencher.iter(|| {
                let mut count = 0u32;
                for target in black_box(&targets).iter() {
                    if array::threshold(black_box(&source), target, 100).is_some() {
                        count += 1;
                    }
                }
                black_box(count)
            })
        });

        // Medium threshold: ~30%
        group.bench_function("within_thresh300_1024b", |bencher| {
            bencher.iter(|| {
                let mut count = 0u32;
                for target in black_box(&targets).iter() {
                    if array::threshold(black_box(&source), target, 300).is_some() {
                        count += 1;
                    }
                }
                black_box(count)
            })
        });

        // Loose threshold: everything passes (worst case for early exit)
        group.bench_function("within_thresh_max_1024b", |bencher| {
            bencher.iter(|| {
                let mut count = 0u32;
                for target in black_box(&targets).iter() {
                    if array::threshold(black_box(&source), target, u32::MAX).is_some() {
                        count += 1;
                    }
                }
                black_box(count)
            })
        });
    }

    // 2048-bit vectors (early exit has more room to save)
    {
        let source: [u8; 256] = random_bytes();
        let targets: Vec<[u8; 256]> = random_bytes_array(BATCH);

        group.bench_function("full_compute_2048b", |bencher| {
            let mut out = vec![0u32; BATCH];
            bencher.iter(|| {
                for (target, dist) in black_box(&targets).iter().zip(out.iter_mut()) {
                    *dist = array::distance(black_box(&source), target);
                }
                black_box(out[0])
            })
        });

        group.bench_function("within_thresh200_2048b", |bencher| {
            bencher.iter(|| {
                let mut count = 0u32;
                for target in black_box(&targets).iter() {
                    if array::threshold(black_box(&source), target, 200).is_some() {
                        count += 1;
                    }
                }
                black_box(count)
            })
        });

        group.bench_function("within_thresh_max_2048b", |bencher| {
            bencher.iter(|| {
                let mut count = 0u32;
                for target in black_box(&targets).iter() {
                    if array::threshold(black_box(&source), target, u32::MAX).is_some() {
                        count += 1;
                    }
                }
                black_box(count)
            })
        });
    }

    group.finish();
}

// ============================================================================
// Streaming top-k simulation: batch_threshold vs plain batch
// ============================================================================

fn streaming_topk_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_topk");

    // Simulate: 10K items scored against 100 interests, keeping top-50
    const NUM_ITEMS: usize = 10_000;
    const NUM_INTERESTS: usize = 100;
    const TOP_K: usize = 50;

    // 1024-bit vectors
    {
        let interests: Vec<[u8; 128]> = random_bytes_array(NUM_INTERESTS);
        let items: Vec<[u8; 128]> = random_bytes_array(NUM_ITEMS);

        // Baseline: array::batch for all, then find min
        group.bench_function("batch_then_min/1024b", |bencher| {
            let mut out = vec![0u32; NUM_INTERESTS];
            bencher.iter(|| {
                let mut heap_threshold = u32::MAX;
                let mut heap_count = 0u32;
                for item in black_box(&items).iter() {
                    array::batch(
                        black_box(item),
                        black_box(&interests),
                        &mut out,
                    );
                    let min_dist = *out.iter().min().unwrap();
                    if heap_count < TOP_K as u32 || min_dist < heap_threshold {
                        if heap_count < TOP_K as u32 {
                            heap_count += 1;
                        }
                        if min_dist < heap_threshold {
                            heap_threshold = min_dist;
                        }
                    }
                }
                black_box(heap_threshold)
            })
        });

        // batch_threshold: same pattern but with early exit
        group.bench_function("batch_threshold/1024b", |bencher| {
            let mut out = vec![0u32; NUM_INTERESTS];
            bencher.iter(|| {
                let mut heap_threshold = u32::MAX;
                let mut heap_count = 0u32;
                for item in black_box(&items).iter() {
                    let best = array::batch_threshold(
                        black_box(item),
                        black_box(&interests),
                        heap_threshold,
                        &mut out,
                    );
                    if heap_count < TOP_K as u32 || best < heap_threshold {
                        if heap_count < TOP_K as u32 {
                            heap_count += 1;
                        }
                        if best < heap_threshold {
                            heap_threshold = best;
                        }
                    }
                }
                black_box(heap_threshold)
            })
        });
    }

    group.finish();
}

criterion_group!(benches, early_exit_benchmarks, streaming_topk_benchmarks);
criterion_main!(benches);

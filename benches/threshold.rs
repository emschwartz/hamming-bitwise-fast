//! Benchmarks for early-exit threshold approaches.
//!
//! These benchmarks use local threshold implementations (not part of the public API)
//! to measure the effectiveness of early-exit strategies.
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
// Local threshold implementation (not part of the public API)
// ============================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const THRESHOLD_BLOCK_SIZE: usize = 64;
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
const THRESHOLD_BLOCK_SIZE: usize = 32;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn distance_impl(a: &[u8], b: &[u8]) -> u32 {
    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);

    let main: u32 = a_chunks
        .clone()
        .zip(b_chunks.clone())
        .map(|(a, b)| {
            let a = u64::from_ne_bytes(a.try_into().unwrap());
            let b = u64::from_ne_bytes(b.try_into().unwrap());
            (a ^ b).count_ones()
        })
        .sum();

    let rem: u32 = a_chunks
        .remainder()
        .iter()
        .zip(b_chunks.remainder())
        .map(|(a, b)| (a ^ b).count_ones())
        .sum();

    main + rem
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline(always)]
fn distance_impl(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

#[inline(always)]
fn threshold_impl(a: &[u8], b: &[u8], threshold: u32) -> Option<u32> {
    let mut distance: u32 = 0;

    let a_blocks = a.chunks_exact(THRESHOLD_BLOCK_SIZE);
    let b_blocks = b.chunks_exact(THRESHOLD_BLOCK_SIZE);
    let a_rem = a_blocks.remainder();
    let b_rem = b_blocks.remainder();

    for (a_block, b_block) in a_blocks.zip(b_blocks) {
        distance += distance_impl(a_block, b_block);
        if distance > threshold {
            return None;
        }
    }

    distance += distance_impl(a_rem, b_rem);

    if distance <= threshold {
        Some(distance)
    } else {
        None
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn opaque_ptr<T>(mut ptr: *const T) -> *const T {
    core::arch::asm!("/* {0} */", inout(reg) ptr, options(nomem, nostack, preserves_flags));
    ptr
}

fn batch_threshold_fn<const N: usize>(
    source: &[u8; N],
    targets: &[[u8; N]],
    max: u32,
    out: &mut [u32],
) -> u32 {
    assert_eq!(targets.len(), out.len());
    let mut best = u32::MAX;
    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let target = unsafe { &*opaque_ptr(target as *const [u8; N]) };
        match threshold_impl(source, target, max) {
            Some(d) => {
                *dist = d;
                if d < best {
                    best = d;
                }
            }
            None => {
                *dist = u32::MAX;
            }
        }
    }
    best
}

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
                    if threshold_impl(black_box(&source), target, 100).is_some() {
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
                    if threshold_impl(black_box(&source), target, 300).is_some() {
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
                    if threshold_impl(black_box(&source), target, black_box(u32::MAX)).is_some() {
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
                    if threshold_impl(black_box(&source), target, 200).is_some() {
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
                    if threshold_impl(black_box(&source), target, black_box(u32::MAX)).is_some() {
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

    // Simulate: 10K items scored against 100 interests, keeping top-50.
    // Note: heap_threshold only decreases (tracking global minimum), which is
    // simpler than a real top-k max-heap (where the threshold is the worst item
    // in the heap and can increase as better items are found). Both benchmark
    // variants use the same simplified logic so the relative comparison is valid.
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
                    let best = batch_threshold_fn(
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

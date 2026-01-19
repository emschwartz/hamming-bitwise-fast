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

fn random_bytes(size: usize) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen()).collect()
}

// ============================================================================
// Group 1A: Type & Loop Style Benchmarks
// ============================================================================

fn bench_type_and_loop_style(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_pair/type_loop_style");

    // Test different embedding sizes: 512, 768, 1024, 2048 bits
    for (name, n_u64s, n_bytes) in [
        ("512bit", 8, 64),
        ("768bit", 12, 96),
        ("1024bit", 16, 128),
        ("2048bit", 32, 256),
    ] {
        group.throughput(Throughput::Elements(1));

        // Byte slice baseline
        let a_bytes = random_bytes(n_bytes);
        let b_bytes = random_bytes(n_bytes);

        group.bench_with_input(
            BenchmarkId::new("slice_baseline", name),
            &(&a_bytes, &b_bytes),
            |bench, (a, b)| {
                bench.iter(|| {
                    hamming_bitwise_fast(
                        criterion::black_box(a),
                        criterion::black_box(b),
                    )
                })
            },
        );

        // Const-generic implementations need macro-based dispatch due to const generics
        match n_u64s {
            8 => bench_const_generic::<8>(&mut group, name),
            12 => bench_const_generic::<12>(&mut group, name),
            16 => bench_const_generic::<16>(&mut group, name),
            32 => bench_const_generic::<32>(&mut group, name),
            _ => unreachable!(),
        }
    }

    group.finish();
}

fn bench_const_generic<const N: usize>(group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>, size_name: &str) {
    let a: Embedding<N> = random_embedding();
    let b: Embedding<N> = random_embedding();

    // Reference + for loop
    group.bench_with_input(
        BenchmarkId::new("ref_for", size_name),
        &(&a, &b),
        |bench, (a, b)| {
            bench.iter(|| {
                hamming_ref_for(
                    criterion::black_box(a),
                    criterion::black_box(b),
                )
            })
        },
    );

    // Reference + iterator
    group.bench_with_input(
        BenchmarkId::new("ref_iter", size_name),
        &(&a, &b),
        |bench, (a, b)| {
            bench.iter(|| {
                hamming_ref_iter(
                    criterion::black_box(a),
                    criterion::black_box(b),
                )
            })
        },
    );

    // Copy + for loop
    group.bench_with_input(
        BenchmarkId::new("copy_for", size_name),
        &(a, b),
        |bench, (a, b)| {
            bench.iter(|| {
                hamming_copy_for(
                    criterion::black_box(*a),
                    criterion::black_box(*b),
                )
            })
        },
    );

    // Copy + iterator
    group.bench_with_input(
        BenchmarkId::new("copy_iter", size_name),
        &(a, b),
        |bench, (a, b)| {
            bench.iter(|| {
                hamming_copy_iter(
                    criterion::black_box(*a),
                    criterion::black_box(*b),
                )
            })
        },
    );
}

// ============================================================================
// Group 1A2: u8 vs u64 Representation
// ============================================================================

fn bench_u8_vs_u64(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_pair/u8_vs_u64");

    // Test with 1024-bit embeddings (128 bytes = 16 u64s)
    const N_BYTES: usize = 128;
    const N_U64S: usize = 16;

    let a_bytes: [u8; N_BYTES] = {
        let mut arr = [0u8; N_BYTES];
        let mut rng = rand::thread_rng();
        for i in 0..N_BYTES {
            arr[i] = rng.gen();
        }
        arr
    };
    let b_bytes: [u8; N_BYTES] = {
        let mut arr = [0u8; N_BYTES];
        let mut rng = rand::thread_rng();
        for i in 0..N_BYTES {
            arr[i] = rng.gen();
        }
        arr
    };

    // Convert to u64 array for comparison
    let a_u64: Embedding<N_U64S> = bytes_to_embedding(&a_bytes);
    let b_u64: Embedding<N_U64S> = bytes_to_embedding(&b_bytes);

    // Also test as slices
    let a_slice: &[u8] = &a_bytes;
    let b_slice: &[u8] = &b_bytes;

    group.throughput(Throughput::Elements(1));

    // --- u8 slice (dynamic) ---
    group.bench_function("u8_slice_dynamic", |bench| {
        bench.iter(|| {
            hamming_bitwise_fast(
                criterion::black_box(a_slice),
                criterion::black_box(b_slice),
            )
        })
    });

    // --- u8 array with for loop ---
    group.bench_function("u8_array_for", |bench| {
        bench.iter(|| {
            hamming_u8_for(
                criterion::black_box(&a_bytes),
                criterion::black_box(&b_bytes),
            )
        })
    });

    // --- u8 array with iterator ---
    group.bench_function("u8_array_iter", |bench| {
        bench.iter(|| {
            hamming_u8_iter(
                criterion::black_box(&a_bytes),
                criterion::black_box(&b_bytes),
            )
        })
    });

    // --- u8 array chunked to u64 ---
    group.bench_function("u8_array_chunked_u64", |bench| {
        bench.iter(|| {
            hamming_u8_chunked(
                criterion::black_box(&a_bytes),
                criterion::black_box(&b_bytes),
            )
        })
    });

    // --- u64 array with for loop ---
    group.bench_function("u64_array_for", |bench| {
        bench.iter(|| {
            hamming_ref_for(
                criterion::black_box(&a_u64),
                criterion::black_box(&b_u64),
            )
        })
    });

    // --- u64 array with iterator ---
    group.bench_function("u64_array_iter", |bench| {
        bench.iter(|| {
            hamming_ref_iter(
                criterion::black_box(&a_u64),
                criterion::black_box(&b_u64),
            )
        })
    });

    // --- u64 with multiversion ---
    #[cfg(feature = "multiversion")]
    group.bench_function("u64_multiversion", |bench| {
        bench.iter(|| {
            hamming_multiversion(
                criterion::black_box(&a_u64),
                criterion::black_box(&b_u64),
            )
        })
    });

    group.finish();
}

// ============================================================================
// Group 1B: Dispatch Strategy Benchmarks
// ============================================================================

fn bench_dispatch_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_pair/dispatch_strategy");

    // Focus on 1024-bit (N=16) as the primary test case
    const N: usize = 16;
    let a: Embedding<N> = random_embedding();
    let b: Embedding<N> = random_embedding();

    group.throughput(Throughput::Elements(1));

    // Auto-vectorized (default compilation)
    // This tests whether the compiler auto-vectorizes without explicit target features
    group.bench_function("auto_vectorized", |bench| {
        bench.iter(|| {
            hamming_ref_iter(
                criterion::black_box(&a),
                criterion::black_box(&b),
            )
        })
    });

    // Multiversion (runtime CPU dispatch)
    #[cfg(feature = "multiversion")]
    group.bench_function("multiversion", |bench| {
        bench.iter(|| {
            hamming_multiversion(
                criterion::black_box(&a),
                criterion::black_box(&b),
            )
        })
    });

    // Note: "native" target is tested by running the entire benchmark with:
    // RUSTFLAGS="-C target-cpu=native" cargo bench --bench single_pair
    // The same "auto_vectorized" benchmark will then use native instructions.
    // Compare results between runs to see the difference.

    group.finish();
}

// ============================================================================
// Group 1C: External Crates Comparison
// ============================================================================

fn bench_external_crates(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_pair/external_crates");

    // Test with 1024-bit embeddings (128 bytes)
    let a_bytes = random_bytes(128);
    let b_bytes = random_bytes(128);
    let a: Embedding<16> = bytes_to_embedding(&a_bytes);
    let b: Embedding<16> = bytes_to_embedding(&b_bytes);

    group.throughput(Throughput::Elements(1));

    // Our best implementation
    group.bench_function("hamming_ref_iter", |bench| {
        bench.iter(|| {
            hamming_ref_iter(
                criterion::black_box(&a),
                criterion::black_box(&b),
            )
        })
    });

    #[cfg(feature = "multiversion")]
    group.bench_function("hamming_multiversion", |bench| {
        bench.iter(|| {
            hamming_multiversion(
                criterion::black_box(&a),
                criterion::black_box(&b),
            )
        })
    });

    // simsimd crate
    group.bench_function("simsimd", |bench| {
        bench.iter(|| {
            simsimd::BinarySimilarity::hamming(
                criterion::black_box(&a_bytes),
                criterion::black_box(&b_bytes),
            )
        })
    });

    // hamming crate
    group.bench_function("hamming_crate", |bench| {
        bench.iter(|| {
            hamming::distance_fast(
                criterion::black_box(&a_bytes),
                criterion::black_box(&b_bytes),
            )
        })
    });

    // triple_accel crate
    group.bench_function("triple_accel", |bench| {
        bench.iter(|| {
            triple_accel::hamming(
                criterion::black_box(&a_bytes),
                criterion::black_box(&b_bytes),
            )
        })
    });

    // hamming_rs crate (x86/x86_64 only)
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    group.bench_function("hamming_rs", |bench| {
        bench.iter(|| {
            hamming_rs::distance_faster(
                criterion::black_box(&a_bytes),
                criterion::black_box(&b_bytes),
            )
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_type_and_loop_style,
    bench_u8_vs_u64,
    bench_dispatch_strategies,
    bench_external_crates
);
criterion_main!(benches);

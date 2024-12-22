use criterion::{criterion_group, criterion_main, Bencher, BenchmarkId, Criterion};
use hamming_bitwise_fast::*;

const BIT_SIZES: [usize; 4] = [512, 768, 1024, 2048];

fn distance_bench<F: 'static + FnMut(&[u8], &[u8]) -> u64>(
    mut f: F,
) -> impl FnMut(&mut Bencher, &usize) {
    move |b, size| {
        let data = vec![0xFF; *size / 8];
        b.iter(|| {
            let d1 = criterion::black_box(&data);
            let d2 = criterion::black_box(&data);
            f(d1, d2)
        })
    }
}

fn bench_hamming(c: &mut Criterion) {
    let mut group = c.benchmark_group("hamming");
    for size in BIT_SIZES {
        group.bench_with_input(
            BenchmarkId::new("hamming-bitwise-fast", size),
            &size,
            distance_bench(|x, y| hamming_bitwise_fast(x, y) as u64),
        );
        group.bench_with_input(
            BenchmarkId::new("hamming-bitwise-fast-16", size),
            &size,
            distance_bench(|x, y| hamming_bitwise_fast_16(x, y) as u64),
        );
        group.bench_with_input(
            BenchmarkId::new("naive", size),
            &size,
            distance_bench(naive_hamming_distance),
        );
        group.bench_with_input(
            BenchmarkId::new("naive_iter", size),
            &size,
            distance_bench(naive_hamming_distance_iter),
        );

        group.bench_with_input(
            BenchmarkId::new("hamming", size),
            &size,
            distance_bench(|x, y| hamming::distance_fast(x, y).unwrap()),
        );
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        group.bench_with_input(
            BenchmarkId::new("hamming_rs", size),
            &size,
            distance_bench(hamming_rs::distance_faster),
        );
        group.bench_with_input(
            BenchmarkId::new("simsimd", size),
            &size,
            distance_bench(|x, y| simsimd::BinarySimilarity::hamming(x, y).unwrap() as u64),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_hamming);
criterion_main!(benches);

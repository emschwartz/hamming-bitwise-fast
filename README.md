# Rust Hamming Distance Benchmark

This benchmarks various Hamming distance implementations in Rust:

- [a naive for-loop based implementation](src/naive.rs)
- [a naive iterator based implementation](src/naive.rs)
- [an auto-vectorized implementation](src/naive.rs)
- [`bitarray`](https://crates.io/crates/bitarray) ![bitarray](https://img.shields.io/crates/d/bitarray)
- [`hamming`](https://crates.io/crates/hamming) ![hamming](https://img.shields.io/crates/d/hamming)
- [`hamming_rs`](https://crates.io/crates/hamming_rs) ![hamming_rs](https://img.shields.io/crates/d/hamming_rs)
- [`simsimd`](https://crates.io/crates/simsimd) ![simsimd](https://img.shields.io/crates/d/simsimd)

## Running the benchmark

```sh
cargo bench
```

Then open the `target/criterion/report/index.html` file in your browser to view the results.

## Results

These were the results running on 3 different types of machines:

### 2023 MacBook Pro M2 Max

![Benchmark results](results/line-chart-macbook.svg)
![Benchmark results](results/violin-chart-macbook.svg)

### Linode 2 CPU 4GB

![Benchmark results](results/line-chart-linode.svg)
![Benchmark results](results/violin-chart-linode.svg)

### Fly.io 2 CPU 4GB

![Benchmark results](results/line-chart-fly.svg)
![Benchmark results](results/violin-chart-fly.svg)

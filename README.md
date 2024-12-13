# Rust Hamming Distance Benchmark

This benchmarks various Hamming distance implementations in Rust:

- [`bitarray`](https://crates.io/crates/bitarray) ![bitarray](https://img.shields.io/crates/d/bitarray)
- [`distances`](https://crates.io/crates/distances) ![distances](https://img.shields.io/crates/d/distances)
- [`hamming`](https://crates.io/crates/hamming) ![hamming](https://img.shields.io/crates/d/hamming)
- [`simsimd`](https://crates.io/crates/simsimd) ![simsimd](https://img.shields.io/crates/d/simsimd)
- [`stringzilla`](https://crates.io/crates/stringzilla) ![stringzilla](https://img.shields.io/crates/d/stringzilla)
- [`triple_accel`](https://crates.io/crates/triple_accel) ![triple_accel](https://img.shields.io/crates/d/triple_accel)

## Running the benchmark

```sh
cargo bench
```

Then open the `target/criterion/report/index.html` file in your browser to view the results.

## Results

These were the results running on 3 different types of machines:

### 2023 MacBook Pro M2 Max

![Benchmark results](results/line-chart-macbook.svg)
![Benchmark results](results/violin-plot-macbook.svg)

### Linode 2 CPU 4GB

![Benchmark results](results/line-chart-linode.svg)
![Benchmark results](results/violin-plot-linode.svg)

### Fly.io 2 CPU 4GB

![Benchmark results](results/line-chart-fly.svg)
![Benchmark results](results/violin-plot-fly.svg)

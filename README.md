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

These were the results running on a 2023 MacBook Pro M2 Max:

1. 🥇 [`simsimd`](https://crates.io/crates/simsimd)
2. 🥈 [the naive implementation!](./src/naive.rs)
3. 🥉 [`hamming`](https://crates.io/crates/hamming)

![Benchmark results](line-chart.svg)

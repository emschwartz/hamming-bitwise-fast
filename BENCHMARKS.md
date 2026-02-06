# Benchmarks

## Table of Contents

- [Benchmark Results](#benchmark-results)
    - [single](#single)
    - [batch](#batch)

## Benchmark Results

### single

|           | `array::distance`                | `slice::distance`                | `simsimd`                       | `hamming_crate`                 | `triple_accel`                   |
|:----------|:---------------------------------|:---------------------------------|:--------------------------------|:--------------------------------|:-------------------------------- |
| **`64`**  | `1.84 ns` (✅ **1.00x**)          | `2.15 ns` (❌ *1.17x slower*)     | `5.11 ns` (❌ *2.77x slower*)    | `8.22 ns` (❌ *4.46x slower*)    | `7.81 ns` (❌ *4.24x slower*)     |
| **`96`**  | `2.30 ns` (✅ **1.00x**)          | `3.73 ns` (❌ *1.62x slower*)     | `8.25 ns` (❌ *3.59x slower*)    | `10.14 ns` (❌ *4.41x slower*)   | `10.56 ns` (❌ *4.59x slower*)    |
| **`128`** | `2.86 ns` (✅ **1.00x**)          | `3.03 ns` (✅ **1.06x slower**)   | `9.66 ns` (❌ *3.38x slower*)    | `13.75 ns` (❌ *4.81x slower*)   | `17.61 ns` (❌ *6.16x slower*)    |
| **`256`** | `4.50 ns` (✅ **1.00x**)          | `4.85 ns` (✅ **1.08x slower**)   | `14.18 ns` (❌ *3.15x slower*)   | `28.22 ns` (❌ *6.27x slower*)   | `24.48 ns` (❌ *5.44x slower*)    |

### batch

|           | `array::batch`                         | `array::distance (loop)`              | `slice::batch`                         | `slice::distance (loop)`              | `simsimd`                       | `hamming_crate`                 | `triple_accel`                   |
|:----------|:---------------------------------------|:--------------------------------------|:---------------------------------------|:--------------------------------------|:--------------------------------|:--------------------------------|:-------------------------------- |
| **`64`**  | `1.27 us` (✅ **1.00x**)                | `1.47 us` (❌ *1.15x slower*)          | `1.88 us` (❌ *1.47x slower*)           | `2.43 us` (❌ *1.91x slower*)          | `5.18 us` (❌ *4.06x slower*)    | `8.31 us` (❌ *6.52x slower*)    | `7.66 us` (❌ *6.01x slower*)     |
| **`96`**  | `1.59 us` (✅ **1.00x**)                | `1.75 us` (✅ **1.10x slower**)        | `3.44 us` (❌ *2.17x slower*)           | `4.02 us` (❌ *2.54x slower*)          | `5.99 us` (❌ *3.78x slower*)    | `10.45 us` (❌ *6.59x slower*)   | `10.85 us` (❌ *6.84x slower*)    |
| **`128`** | `2.29 us` (✅ **1.00x**)                | `2.67 us` (❌ *1.16x slower*)          | `2.84 us` (❌ *1.24x slower*)           | `3.54 us` (❌ *1.54x slower*)          | `7.18 us` (❌ *3.13x slower*)    | `12.51 us` (❌ *5.46x slower*)   | `12.51 us` (❌ *5.46x slower*)    |
| **`256`** | `4.73 us` (✅ **1.00x**)                | `5.02 us` (✅ **1.06x slower**)        | `5.07 us` (✅ **1.07x slower**)         | `6.16 us` (❌ *1.30x slower*)          | `10.93 us` (❌ *2.31x slower*)   | `19.55 us` (❌ *4.13x slower*)   | `21.91 us` (❌ *4.63x slower*)    |

**Note on simsimd:** `simsimd` returns a normalized `f64` (distance / total bits), so it performs additional floating-point division compared to the other implementations which return raw integer counts.

---
Made with [criterion-table](https://github.com/nu11ptr/criterion-table)


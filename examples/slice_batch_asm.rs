//! Example for inspecting slice batch assembly.
//!
//! Build and emit assembly with:
//!   cargo rustc --release --example slice_batch_asm --features multiversion_x86 \
//!     --target x86_64-unknown-linux-gnu -- --emit asm
//!
//! Output will be at: target/x86_64-unknown-linux-gnu/release/examples/slice_batch_asm.s

use hamming_bitwise_fast::slice;
use std::hint::black_box;

#[inline(never)]
pub fn bench_slice_batch(source: &[u8], targets: &[&[u8]], out: &mut [u32]) {
    slice::batch(source, targets, out);
}

fn main() {
    // 1024-bit arrays (128 bytes)
    let source: Vec<u8> = black_box(vec![0x55; 128]);
    let targets_owned: Vec<Vec<u8>> = black_box(vec![vec![0xAA; 128]; 64]);
    let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();
    let mut out = vec![0u32; 64];

    // Call batch function
    bench_slice_batch(black_box(&source), black_box(&targets), black_box(&mut out));

    // Prevent optimization
    println!("Batch result[0]: {}", black_box(out[0]));
}

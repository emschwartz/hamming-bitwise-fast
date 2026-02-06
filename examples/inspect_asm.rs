//! Example for inspecting generated assembly.
//!
//! Build with:
//!   RUSTFLAGS="-C target-cpu=x86-64-v4" cargo build --release --example inspect_asm
//!
//! Then inspect:
//!   objdump -d target/release/examples/inspect_asm | less
//!
//! Or on macOS:
//!   objdump -d target/release/examples/inspect_asm | less

use hamming_bitwise_fast::array;
use std::hint::black_box;

fn main() {
    // 1024-bit arrays (128 bytes)
    let source: [u8; 128] = black_box([0x55; 128]);
    let targets: Vec<[u8; 128]> = black_box(vec![[0xAA; 128]; 64]);
    let mut out = vec![0u32; 64];

    // Call batch function
    array::batch(black_box(&source), black_box(&targets), black_box(&mut out));

    // Prevent optimization
    println!("Batch result[0]: {}", black_box(out[0]));

    // Call single function in a loop for comparison
    let mut loop_out = vec![0u32; 64];
    for (target, dist) in targets.iter().zip(loop_out.iter_mut()) {
        *dist = array::distance(black_box(&source), black_box(target));
    }

    println!("Loop result[0]: {}", black_box(loop_out[0]));

    // Verify correctness
    assert_eq!(out, loop_out, "Batch and loop results should match");

    // Expected: each byte differs in all 8 bits (0x55 ^ 0xAA = 0xFF)
    // So 128 bytes * 8 bits = 1024 bits different
    assert_eq!(out[0], 1024, "Expected 1024 bits different");

    println!("All results correct!");
}

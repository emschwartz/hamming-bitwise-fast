//! Example for checking generated assembly.
//!
//! Build with: cargo build --example asm_check --release
//! Check assembly with: cargo asm --example asm_check check_hamming

use hamming_bitwise_fast::{hamming, hamming_batch, hamming_bitwise_fast};

#[no_mangle]
pub fn check_bitwise_fast(a: &[u8], b: &[u8]) -> u32 {
    hamming_bitwise_fast(a, b)
}

#[no_mangle]
pub fn check_hamming(a: &[u64; 16], b: &[u64; 16]) -> u32 {
    hamming(a, b)
}

#[no_mangle]
pub fn check_hamming_batch(source: &[u64; 16], targets: &[[u64; 16]], out: &mut [u32]) {
    hamming_batch(source, targets, out)
}

fn main() {
    let a = [0u64; 16];
    let b = [u64::MAX; 16];
    println!("hamming distance: {}", check_hamming(&a, &b));
}

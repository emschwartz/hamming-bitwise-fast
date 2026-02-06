//! Check actual library assembly.
//! Run: cargo build --release --example lib_asm
//! Then: otool -tV target/release/examples/lib_asm | grep -A 80 'lib_check'

use hamming_bitwise_fast::{array, slice};
use std::hint::black_box;

/// Wrapper to force a symbol for the array function
#[inline(never)]
pub fn lib_check_array_128(a: &[u8; 128], b: &[u8; 128]) -> u32 {
    array::distance(a, b)
}

/// Wrapper for slice function
#[inline(never)]
pub fn lib_check_slice(a: &[u8], b: &[u8]) -> u32 {
    slice::distance(a, b)
}

fn main() {
    let a_arr: [u8; 128] = [0; 128];
    let b_arr: [u8; 128] = [0xFF; 128];

    println!("array: {}", lib_check_array_128(black_box(&a_arr), black_box(&b_arr)));

    let a_slice = vec![0u8; 128];
    let b_slice = vec![0xFFu8; 128];
    println!("slice: {}", lib_check_slice(black_box(&a_slice), black_box(&b_slice)));
}

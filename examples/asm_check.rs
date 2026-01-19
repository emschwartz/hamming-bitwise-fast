//! Example for checking generated assembly
use hamming_bitwise_fast::*;

#[no_mangle]
pub fn check_slice_assert_aligned(a: &[u8], b: &[u8]) -> u32 {
    hamming_slice_assert_aligned(a, b)
}

#[no_mangle]
pub fn check_slice_assert_u64_chunks(a: &[u8], b: &[u8]) -> u32 {
    hamming_slice_assert_u64_chunks(a, b)
}

#[no_mangle]
pub fn check_u8_for(a: &[u8; 128], b: &[u8; 128]) -> u32 {
    hamming_u8_for(a, b)
}

#[no_mangle]
pub fn check_u64_iter(a: &[u64; 16], b: &[u64; 16]) -> u32 {
    hamming_ref_iter(a, b)
}

#[no_mangle]
pub fn check_bitwise_fast(a: &[u8], b: &[u8]) -> u32 {
    hamming_bitwise_fast(a, b)
}

fn main() {
    let a = [0u8; 128];
    let b = [0u8; 128];
    println!("{}", check_slice_assert_aligned(&a, &b));
}

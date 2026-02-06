// Examples for checking assembly optimization of batch functions
use hamming_bitwise_fast::{array, slice};

// Single array distance
#[no_mangle]
#[inline(never)]
pub fn hamming_128(a: &[u8; 128], b: &[u8; 128]) -> u32 {
    array::distance(a, b)
}

// Array threshold with u32::MAX (should compile identically to hamming_128)
#[no_mangle]
#[inline(never)]
pub fn hamming_128_threshold_max(a: &[u8; 128], b: &[u8; 128]) -> u32 {
    array::threshold(a, b, u32::MAX).unwrap()
}

// Array threshold with a real threshold
#[no_mangle]
#[inline(never)]
pub fn hamming_128_threshold(a: &[u8; 128], b: &[u8; 128], max: u32) -> Option<u32> {
    array::threshold(a, b, max)
}

// Array batch
#[no_mangle]
#[inline(never)]
pub fn array_batch_128(source: &[u8; 128], targets: &[[u8; 128]], out: &mut [u32]) {
    array::batch(source, targets, out);
}

// Slice batch
#[no_mangle]
#[inline(never)]
pub fn slice_batch(source: &[u8], targets: &[&[u8]], out: &mut [u32]) {
    slice::batch(source, targets, out);
}

// Small array distance (8 bytes = 64 bits)
#[no_mangle]
#[inline(never)]
pub fn hamming_8(a: &[u8; 8], b: &[u8; 8]) -> u32 {
    array::distance(a, b)
}

fn main() {
    let a = [0u8; 128];
    let b = [0xFFu8; 128];
    println!("{}", hamming_128(&a, &b));
    println!("{}", hamming_128_threshold_max(&a, &b));
    println!("{:?}", hamming_128_threshold(&a, &b, 100));
    let targets = vec![[0xFFu8; 128]; 2];
    let mut out = vec![0u32; 2];
    array_batch_128(&a, &targets, &mut out);
    println!("{:?}", out);
    let a8 = [0u8; 8];
    let b8 = [0xFFu8; 8];
    println!("{}", hamming_8(&a8, &b8));
}

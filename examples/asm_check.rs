// Examples for checking assembly optimization of batch functions
use hamming_bitwise_fast::{array, slice};

// Single array function for comparison
#[no_mangle]
pub fn hamming_128(a: &[u8; 128], b: &[u8; 128]) -> u32 {
    array::distance(a, b)
}

// Array batch - currently inlines chunking logic on x86
#[no_mangle]
pub fn array_batch_128(source: &[u8; 128], targets: &[[u8; 128]], out: &mut [u32]) {
    array::batch(source, targets, out);
}

// Slice batch - currently inlines chunking logic on x86
#[no_mangle]
pub fn slice_batch(source: &[u8], targets: &[&[u8]], out: &mut [u32]) {
    slice::batch(source, targets, out);
}

fn main() {
    let a = [0u8; 128];
    let b = [0xFFu8; 128];
    println!("{}", hamming_128(&a, &b));
}

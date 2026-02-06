//! Assembly inspection wrappers for threshold APIs.
//!
//! Build: RUSTFLAGS="-C target-cpu=native" cargo build --release --example threshold_asm
//! View (ARM): otool -tV target/release/examples/threshold_asm | grep -A 80 'within'
//! View (x86):  objdump -d target/release/examples/threshold_asm | grep -A 80 'within'

use std::hint::black_box;

use hamming_bitwise_fast::array;

#[no_mangle]
#[inline(never)]
pub fn within_128(a: &[u8; 128], b: &[u8; 128], threshold: u32) -> Option<u32> {
    array::threshold(a, b, threshold)
}

fn main() {
    let a: [u8; 128] = [0; 128];
    let b: [u8; 128] = [0xFF; 128];
    println!(
        "within(100): {:?}",
        within_128(black_box(&a), black_box(&b), 100)
    );
    println!(
        "within(2000): {:?}",
        within_128(black_box(&a), black_box(&b), 2000)
    );
}

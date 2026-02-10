use test_case::test_case;

use super::{array, slice};

// ── Slice distance ──────────────────────────────────────────────────

#[test_case(7 ; "7 bytes")]
#[test_case(13 ; "13 bytes")]
#[test_case(64 ; "64 bytes")]
#[test_case(96 ; "96 bytes")]
#[test_case(100 ; "100 bytes")]
#[test_case(128 ; "128 bytes")]
#[test_case(256 ; "256 bytes")]
fn slice_distance_all_bits_different(size: usize) {
    let a = vec![0u8; size];
    let b = vec![0xFFu8; size];
    assert_eq!(slice::distance(&a, &b), (size * 8) as u32);
}

#[test_case(7 ; "7 bytes")]
#[test_case(64 ; "64 bytes")]
#[test_case(128 ; "128 bytes")]
fn slice_distance_identical(size: usize) {
    let a = vec![0u8; size];
    assert_eq!(slice::distance(&a, &a), 0);
}

#[test_case(7 ; "7 bytes")]
#[test_case(128 ; "128 bytes")]
fn slice_distance_single_bit(size: usize) {
    let a = vec![0u8; size];
    let mut b = vec![0u8; size];
    b[0] = 1;
    assert_eq!(slice::distance(&a, &b), 1);
}

// ── Array distance ──────────────────────────────────────────────────

#[test_case(0xFF, 1024 ; "all bits different")]
#[test_case(0x00, 0 ; "identical")]
#[test_case(0x01, 128 ; "one bit per byte")]
fn array_distance_128(fill_b: u8, expected: u32) {
    let a = [0u8; 128];
    let b = [fill_b; 128];
    assert_eq!(array::distance(&a, &b), expected);
}

#[test]
fn array_distance_single_bit() {
    let mut a = [0u8; 128];
    let b = [0u8; 128];
    a[0] = 1;
    assert_eq!(array::distance(&a, &b), 1);
}

// ── Array matches slice ─────────────────────────────────────────────

#[test_case(128 ; "128 bytes")]
fn array_matches_slice(_size: usize) {
    let a: [u8; 128] = std::array::from_fn(|i| i as u8);
    let b: [u8; 128] = std::array::from_fn(|i| (i + 128) as u8);
    assert_eq!(array::distance(&a, &b), slice::distance(&a, &b));
}

// ── Batch: array ────────────────────────────────────────────────────

#[test_case(0xFF, 1024 ; "all bits different")]
#[test_case(0x00, 0 ; "identical")]
#[test_case(0x01, 128 ; "one bit per byte")]
fn array_batch_128(fill: u8, expected: u32) {
    let source = [0u8; 128];
    let targets = [[fill; 128]];
    let mut out = [0u32; 1];
    array::batch(&source, &targets, &mut out);
    assert_eq!(out[0], expected);
}

#[test_case(0xFF, 104 ; "all bits different")]
#[test_case(0x00, 0 ; "identical")]
fn array_batch_13(fill: u8, expected: u32) {
    let source = [0u8; 13];
    let targets = [[fill; 13]];
    let mut out = [0u32; 1];
    array::batch(&source, &targets, &mut out);
    assert_eq!(out[0], expected);
}

#[test_case(3 ; "3 targets")]
fn array_batch_multiple_targets(num_targets: usize) {
    let source = [0u8; 128];
    let targets = [[0xFFu8; 128], [0u8; 128], [1u8; 128]];
    let mut out = [0u32; 3];
    array::batch(&source, &targets[..num_targets], &mut out[..num_targets]);
    assert_eq!(out[..num_targets], [1024, 0, 128][..num_targets]);
}

// ── Batch: slice ────────────────────────────────────────────────────

#[test_case(0xFF, 1024 ; "all bits different")]
#[test_case(0x00, 0 ; "identical")]
#[test_case(0x01, 128 ; "one bit per byte")]
fn slice_batch_128(fill: u8, expected: u32) {
    let source = vec![0u8; 128];
    let target = vec![fill; 128];
    let targets: Vec<&[u8]> = vec![target.as_slice()];
    let mut out = vec![0u32; 1];
    slice::batch(&source, &targets, &mut out);
    assert_eq!(out[0], expected);
    assert_eq!(slice::distance(&source, &target), expected);
}

#[test_case(3 ; "3 targets")]
fn slice_batch_multiple_targets(num_targets: usize) {
    let source = vec![0u8; 128];
    let t1 = vec![0xFFu8; 128];
    let t2 = vec![0u8; 128];
    let t3 = vec![1u8; 128];
    let all = vec![t1, t2, t3];
    let targets: Vec<&[u8]> = all[..num_targets].iter().map(|v| v.as_slice()).collect();
    let mut out = vec![0u32; num_targets];
    slice::batch(&source, &targets, &mut out);
    assert_eq!(out, [1024, 0, 128][..num_targets]);
    for (i, target) in targets.iter().enumerate() {
        assert_eq!(out[i], slice::distance(&source, target));
    }
}

// ── Edge cases ──────────────────────────────────────────────────────

#[test_case(vec![] ; "empty targets")]
fn slice_batch_empty(targets: Vec<&[u8]>) {
    let source = vec![0u8; 128];
    let mut out: Vec<u32> = vec![];
    slice::batch(&source, &targets, &mut out);
}

#[test_case(2, 1 ; "output too small")]
#[test_case(2, 3 ; "output too large")]
#[should_panic]
fn slice_batch_output_size_mismatch(num_targets: usize, out_size: usize) {
    let source = vec![0u8; 128];
    let targets_owned: Vec<Vec<u8>> = (0..num_targets).map(|_| vec![0u8; 128]).collect();
    let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();
    let mut out = vec![0u32; out_size];
    slice::batch(&source, &targets, &mut out);
}

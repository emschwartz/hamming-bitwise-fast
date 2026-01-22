use super::*;

#[test]
fn hamming_bitwise_slice_correctness() {
    let a = [0u8; 128];
    let b = [0xFFu8; 128];
    assert_eq!(hamming_bitwise_slice(&a, &b), 1024);
    assert_eq!(hamming_bitwise_slice(&a, &a), 0);

    let mut c = [0u8; 128];
    c[0] = 1;
    assert_eq!(hamming_bitwise_slice(&a, &c), 1);
}

#[test]
fn hamming_bitwise_array_correctness() {
    let a: [u8; 128] = [0; 128];
    let b: [u8; 128] = [0xFF; 128];
    assert_eq!(hamming_bitwise_array(&a, &b), 1024);
    assert_eq!(hamming_bitwise_array(&a, &a), 0);

    let mut c = [0u8; 128];
    c[0] = 1;
    assert_eq!(hamming_bitwise_array(&a, &c), 1);
}

#[test]
fn hamming_bitwise_array_batch_correctness() {
    let source: [u8; 128] = [0; 128];
    let targets = vec![
        [0xFFu8; 128], // 1024 bits different
        [0u8; 128],    // 0 bits different
        [1u8; 128],    // 128 bits different (one bit per byte)
    ];
    let mut out = vec![0u32; 3];

    hamming_bitwise_array_batch(&source, &targets, &mut out);

    assert_eq!(out[0], 1024);
    assert_eq!(out[1], 0);
    assert_eq!(out[2], 128);
}

#[test]
fn hamming_bitwise_array_matches_slice() {
    let a: [u8; 128] = std::array::from_fn(|i| i as u8);
    let b: [u8; 128] = std::array::from_fn(|i| (i + 128) as u8);

    assert_eq!(hamming_bitwise_array(&a, &b), hamming_bitwise_slice(&a, &b));
}

#[test]
fn different_embedding_sizes() {
    // 512-bit (64 bytes)
    let a: [u8; 64] = [0; 64];
    let b: [u8; 64] = [0xFF; 64];
    assert_eq!(hamming_bitwise_array(&a, &b), 512);

    // 768-bit (96 bytes)
    let a: [u8; 96] = [0; 96];
    let b: [u8; 96] = [0xFF; 96];
    assert_eq!(hamming_bitwise_array(&a, &b), 768);

    // 2048-bit (256 bytes)
    let a: [u8; 256] = [0; 256];
    let b: [u8; 256] = [0xFF; 256];
    assert_eq!(hamming_bitwise_array(&a, &b), 2048);
}

#[test]
fn odd_sizes_with_remainder() {
    // 7 bytes (not a multiple of 8) - tests remainder handling
    let a: [u8; 7] = [0; 7];
    let b: [u8; 7] = [0xFF; 7];
    assert_eq!(hamming_bitwise_array(&a, &b), 56); // 7 * 8 = 56 bits

    // 13 bytes (8 + 5 remainder)
    let a: [u8; 13] = [0; 13];
    let b: [u8; 13] = [0xFF; 13];
    assert_eq!(hamming_bitwise_array(&a, &b), 104); // 13 * 8 = 104 bits

    // 100 bytes (96 + 4 remainder)
    let a: [u8; 100] = [0; 100];
    let b: [u8; 100] = [0xFF; 100];
    assert_eq!(hamming_bitwise_array(&a, &b), 800); // 100 * 8 = 800 bits

    // Also test batch with odd size
    let source: [u8; 13] = [0; 13];
    let targets = vec![[0xFFu8; 13], [0u8; 13]];
    let mut out = vec![0u32; 2];
    hamming_bitwise_array_batch(&source, &targets, &mut out);
    assert_eq!(out[0], 104);
    assert_eq!(out[1], 0);
}

#[test]
fn hamming_bitwise_slice_batch_correctness() {
    let source = vec![0u8; 128];
    let targets_owned = vec![
        vec![0xFFu8; 128], // 1024 bits different
        vec![0u8; 128],    // 0 bits different
        vec![1u8; 128],    // 128 bits different (one bit per byte)
    ];
    let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();
    let mut out = vec![0u32; 3];

    hamming_bitwise_slice_batch(&source, &targets, &mut out);

    assert_eq!(out[0], 1024);
    assert_eq!(out[1], 0);
    assert_eq!(out[2], 128);

    // Verify results match individual slice calls
    for (i, target) in targets.iter().enumerate() {
        assert_eq!(out[i], hamming_bitwise_slice(&source, target));
    }
}

#[test]
fn hamming_bitwise_slice_batch_empty() {
    let source = vec![0u8; 128];
    let targets: Vec<&[u8]> = vec![];
    let mut out: Vec<u32> = vec![];

    // Should succeed with empty inputs
    hamming_bitwise_slice_batch(&source, &targets, &mut out);
}

#[test]
#[should_panic]
fn hamming_bitwise_slice_batch_output_size_mismatch() {
    let source = vec![0u8; 128];
    let targets_owned = vec![vec![0u8; 128], vec![0u8; 128]];
    let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();
    let mut out = vec![0u32; 1]; // Wrong size!

    hamming_bitwise_slice_batch(&source, &targets, &mut out);
}

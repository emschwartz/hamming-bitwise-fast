use super::{array, slice};

#[test]
fn slice_distance_correctness() {
    let a = [0u8; 128];
    let b = [0xFFu8; 128];
    assert_eq!(slice::distance(&a, &b), 1024);
    assert_eq!(slice::distance(&a, &a), 0);

    let mut c = [0u8; 128];
    c[0] = 1;
    assert_eq!(slice::distance(&a, &c), 1);
}

#[test]
fn array_distance_correctness() {
    let a: [u8; 128] = [0; 128];
    let b: [u8; 128] = [0xFF; 128];
    assert_eq!(array::distance(&a, &b), 1024);
    assert_eq!(array::distance(&a, &a), 0);

    let mut c = [0u8; 128];
    c[0] = 1;
    assert_eq!(array::distance(&a, &c), 1);
}

#[test]
fn array_batch_correctness() {
    let source: [u8; 128] = [0; 128];
    let targets = vec![
        [0xFFu8; 128], // 1024 bits different
        [0u8; 128],    // 0 bits different
        [1u8; 128],    // 128 bits different (one bit per byte)
    ];
    let mut out = vec![0u32; 3];

    array::batch(&source, &targets, &mut out);

    assert_eq!(out[0], 1024);
    assert_eq!(out[1], 0);
    assert_eq!(out[2], 128);
}

#[test]
fn array_matches_slice() {
    let a: [u8; 128] = std::array::from_fn(|i| i as u8);
    let b: [u8; 128] = std::array::from_fn(|i| (i + 128) as u8);

    assert_eq!(array::distance(&a, &b), slice::distance(&a, &b));
}

#[test]
fn different_embedding_sizes() {
    // 512-bit (64 bytes)
    let a: [u8; 64] = [0; 64];
    let b: [u8; 64] = [0xFF; 64];
    assert_eq!(array::distance(&a, &b), 512);

    // 768-bit (96 bytes)
    let a: [u8; 96] = [0; 96];
    let b: [u8; 96] = [0xFF; 96];
    assert_eq!(array::distance(&a, &b), 768);

    // 2048-bit (256 bytes)
    let a: [u8; 256] = [0; 256];
    let b: [u8; 256] = [0xFF; 256];
    assert_eq!(array::distance(&a, &b), 2048);
}

#[test]
fn odd_sizes_with_remainder() {
    // 7 bytes (not a multiple of 8) - tests remainder handling
    let a: [u8; 7] = [0; 7];
    let b: [u8; 7] = [0xFF; 7];
    assert_eq!(array::distance(&a, &b), 56); // 7 * 8 = 56 bits

    // 13 bytes (8 + 5 remainder)
    let a: [u8; 13] = [0; 13];
    let b: [u8; 13] = [0xFF; 13];
    assert_eq!(array::distance(&a, &b), 104); // 13 * 8 = 104 bits

    // 100 bytes (96 + 4 remainder)
    let a: [u8; 100] = [0; 100];
    let b: [u8; 100] = [0xFF; 100];
    assert_eq!(array::distance(&a, &b), 800); // 100 * 8 = 800 bits

    // Also test batch with odd size
    let source: [u8; 13] = [0; 13];
    let targets = vec![[0xFFu8; 13], [0u8; 13]];
    let mut out = vec![0u32; 2];
    array::batch(&source, &targets, &mut out);
    assert_eq!(out[0], 104);
    assert_eq!(out[1], 0);
}

#[test]
fn slice_batch_correctness() {
    let source = vec![0u8; 128];
    let targets_owned = vec![
        vec![0xFFu8; 128], // 1024 bits different
        vec![0u8; 128],    // 0 bits different
        vec![1u8; 128],    // 128 bits different (one bit per byte)
    ];
    let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();
    let mut out = vec![0u32; 3];

    slice::batch(&source, &targets, &mut out);

    assert_eq!(out[0], 1024);
    assert_eq!(out[1], 0);
    assert_eq!(out[2], 128);

    // Verify results match individual slice calls
    for (i, target) in targets.iter().enumerate() {
        assert_eq!(out[i], slice::distance(&source, target));
    }
}

#[test]
fn slice_batch_empty() {
    let source = vec![0u8; 128];
    let targets: Vec<&[u8]> = vec![];
    let mut out: Vec<u32> = vec![];

    // Should succeed with empty inputs
    slice::batch(&source, &targets, &mut out);
}

#[test]
#[should_panic]
fn slice_batch_output_size_mismatch() {
    let source = vec![0u8; 128];
    let targets_owned = vec![vec![0u8; 128], vec![0u8; 128]];
    let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();
    let mut out = vec![0u32; 1]; // Wrong size!

    slice::batch(&source, &targets, &mut out);
}

// ============================================================================
// Array threshold tests
// ============================================================================

#[test]
fn threshold_within_returns_some() {
    let a: [u8; 128] = [0; 128];
    let mut b: [u8; 128] = [0; 128];
    b[0] = 0xFF; // 8 bits different

    assert_eq!(array::threshold(&a, &b, 10), Some(8));
    assert_eq!(array::threshold(&a, &b, 8), Some(8));
}

#[test]
fn threshold_exceeds_returns_none() {
    let a: [u8; 128] = [0; 128];
    let b: [u8; 128] = [0xFF; 128]; // 1024 bits different

    assert_eq!(array::threshold(&a, &b, 100), None);
    assert_eq!(array::threshold(&a, &b, 0), None);
}

#[test]
fn threshold_exact_boundary() {
    let a: [u8; 128] = [0; 128];
    let b: [u8; 128] = [0xFF; 128]; // 1024 bits

    // At exact distance: should return Some
    assert_eq!(array::threshold(&a, &b, 1024), Some(1024));
    // One below: should return None
    assert_eq!(array::threshold(&a, &b, 1023), None);
}

#[test]
fn threshold_zero_distance() {
    let a: [u8; 128] = [0; 128];
    assert_eq!(array::threshold(&a, &a, 0), Some(0));
}

#[test]
fn threshold_matches_distance() {
    // When threshold is large enough, result should match array::distance
    let a: [u8; 128] = std::array::from_fn(|i| i as u8);
    let b: [u8; 128] = std::array::from_fn(|i| (i + 128) as u8);
    let expected = array::distance(&a, &b);
    assert_eq!(array::threshold(&a, &b, u32::MAX), Some(expected));
}

#[test]
fn threshold_small_arrays() {
    // Test with arrays smaller than 64 bytes (no full check interval)
    let a: [u8; 7] = [0; 7];
    let b: [u8; 7] = [0xFF; 7];
    assert_eq!(array::threshold(&a, &b, 56), Some(56));
    assert_eq!(array::threshold(&a, &b, 55), None);
}

// ============================================================================
// Slice threshold tests
// ============================================================================

#[test]
fn slice_threshold_within_returns_some() {
    let a = [0u8; 128];
    let mut b = [0u8; 128];
    b[0] = 0xFF;

    assert_eq!(slice::threshold(&a, &b, 10), Some(8));
    assert_eq!(slice::threshold(&a, &b, 8), Some(8));
}

#[test]
fn slice_threshold_exceeds_returns_none() {
    let a = [0u8; 128];
    let b = [0xFFu8; 128];

    assert_eq!(slice::threshold(&a, &b, 100), None);
    assert_eq!(slice::threshold(&a, &b, 0), None);
}

#[test]
fn slice_threshold_matches_array_threshold() {
    let a: [u8; 128] = std::array::from_fn(|i| i as u8);
    let b: [u8; 128] = std::array::from_fn(|i| (i + 128) as u8);

    for max in [0, 100, 400, 600, 1024, u32::MAX] {
        assert_eq!(
            slice::threshold(&a, &b, max),
            array::threshold(&a, &b, max),
            "mismatch at max={}",
            max
        );
    }
}

// ============================================================================
// Array batch_threshold tests
// ============================================================================

#[test]
fn batch_threshold_correctness() {
    let source: [u8; 128] = [0; 128];
    let targets = vec![
        [0xFFu8; 128], // 1024 bits — should be rejected at threshold 500
        [0u8; 128],    // 0 bits — within threshold
        [1u8; 128],    // 128 bits — within threshold
    ];
    let mut out = vec![0u32; 3];

    let best = array::batch_threshold(&source, &targets, 500, &mut out);

    assert_eq!(out[0], u32::MAX); // rejected
    assert_eq!(out[1], 0);
    assert_eq!(out[2], 128);
    assert_eq!(best, 0);
}

#[test]
fn batch_threshold_all_rejected() {
    let source: [u8; 128] = [0; 128];
    let targets = vec![[0xFFu8; 128]; 5]; // all 1024 bits away
    let mut out = vec![0u32; 5];

    let best = array::batch_threshold(&source, &targets, 100, &mut out);

    assert!(out.iter().all(|&d| d == u32::MAX));
    assert_eq!(best, u32::MAX);
}

#[test]
fn batch_threshold_all_pass() {
    let source: [u8; 128] = [0; 128];
    let targets = vec![[0u8; 128], [1u8; 128]];
    let mut out = vec![0u32; 2];

    let best = array::batch_threshold(&source, &targets, u32::MAX, &mut out);

    assert_eq!(out[0], 0);
    assert_eq!(out[1], 128);
    assert_eq!(best, 0);
}

#[test]
fn batch_threshold_returns_minimum() {
    let source: [u8; 128] = [0; 128];
    let mut close = [0u8; 128];
    close[0] = 1; // 1 bit different
    let mut medium = [0u8; 128];
    medium[0] = 0xFF; // 8 bits different

    let targets = vec![[0xFFu8; 128], medium, close];
    let mut out = vec![0u32; 3];

    let best = array::batch_threshold(&source, &targets, 500, &mut out);

    assert_eq!(out[0], u32::MAX); // 1024, rejected
    assert_eq!(out[1], 8);
    assert_eq!(out[2], 1);
    assert_eq!(best, 1); // minimum of accepted
}

#[test]
fn batch_threshold_matches_individual_calls() {
    let source: [u8; 128] = std::array::from_fn(|i| i as u8);
    let targets: Vec<[u8; 128]> = (0..10)
        .map(|offset| std::array::from_fn(|i| (i.wrapping_mul(offset + 1)) as u8))
        .collect();
    let max = 400;

    // Individual calls
    let expected: Vec<u32> = targets
        .iter()
        .map(|t| array::threshold(&source, t, max).unwrap_or(u32::MAX))
        .collect();

    // Batch call
    let mut actual = vec![0u32; 10];
    array::batch_threshold(&source, &targets, max, &mut actual);

    assert_eq!(actual, expected);
}

// ============================================================================
// Slice batch_threshold tests
// ============================================================================

#[test]
fn slice_batch_threshold_correctness() {
    let source = vec![0u8; 128];
    let targets_owned = vec![vec![0xFFu8; 128], vec![0u8; 128], vec![1u8; 128]];
    let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();
    let mut out = vec![0u32; 3];

    let best = slice::batch_threshold(&source, &targets, 500, &mut out);

    assert_eq!(out[0], u32::MAX);
    assert_eq!(out[1], 0);
    assert_eq!(out[2], 128);
    assert_eq!(best, 0);
}

#[test]
fn slice_batch_threshold_matches_array() {
    let source: [u8; 128] = std::array::from_fn(|i| i as u8);
    let targets: Vec<[u8; 128]> = (0..10)
        .map(|offset| std::array::from_fn(|i| (i.wrapping_mul(offset + 1)) as u8))
        .collect();
    let max = 400;

    // Array batch_threshold
    let mut array_out = vec![0u32; 10];
    array::batch_threshold(&source, &targets, max, &mut array_out);

    // Slice batch_threshold
    let targets_refs: Vec<&[u8]> = targets.iter().map(|a| a.as_slice()).collect();
    let mut slice_out = vec![0u32; 10];
    slice::batch_threshold(&source[..], &targets_refs, max, &mut slice_out);

    assert_eq!(array_out, slice_out);
}

#[inline]
pub fn naive_hamming_distance(x: &[u8], y: &[u8]) -> u64 {
    assert_eq!(x.len(), y.len());
    let mut distance: u32 = 0;
    for i in 0..x.len() {
        distance += (x[i] ^ y[i]).count_ones();
    }
    distance as u64
}

#[inline]
pub fn naive_hamming_distance_iter(x: &[u8], y: &[u8]) -> u64 {
    assert_eq!(x.len(), y.len());
    x.iter()
        .zip(y)
        .fold(0, |a, (b, c)| a + (*b ^ *c).count_ones()) as u64
}

#[inline]
pub fn hamming_distance_auto_vectorized(x: &[u8], y: &[u8]) -> u64 {
    assert_eq!(x.len(), y.len());
    let mut distance: u32 = 0;

    // Process 8 bytes at a time using u64
    for (x_chunk, y_chunk) in x.chunks_exact(8).zip(y.chunks_exact(8)) {
        let x_val = u64::from_ne_bytes(x_chunk.try_into().unwrap());
        let y_val = u64::from_ne_bytes(y_chunk.try_into().unwrap());
        distance += (x_val ^ y_val).count_ones();
    }

    // Handle remaining bytes
    for (&x_byte, &y_byte) in x
        .chunks_exact(8)
        .remainder()
        .iter()
        .zip(y.chunks_exact(8).remainder())
    {
        distance += (x_byte ^ y_byte).count_ones();
    }

    distance as u64
}

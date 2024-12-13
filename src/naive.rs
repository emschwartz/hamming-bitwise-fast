pub fn naive_hamming_distance(x: &[u8], y: &[u8]) -> u64 {
    assert_eq!(x.len(), y.len());
    let mut distance = 0;
    for i in 0..x.len() {
        distance += (x[i] ^ y[i]).count_ones() as u64;
    }
    distance
}

pub fn naive_hamming_distance_iter(x: &[u8], y: &[u8]) -> u64 {
    assert_eq!(x.len(), y.len());
    x.iter()
        .zip(y)
        .fold(0, |a, (b, c)| a + (*b ^ *c).count_ones() as u64)
}

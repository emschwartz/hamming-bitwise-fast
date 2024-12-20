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
    x.iter()
        .zip(y)
        .fold(0, |a, (b, c)| a + (*b ^ *c).count_ones()) as u64
}

#[inline]
pub fn hamming_distance_auto_vectorized(x: &[u8], y: &[u8]) -> u64 {
    assert_eq!(x.len(), y.len());

    // Process 8 bytes at a time using u64
    let mut distance = x
        .chunks_exact(8)
        .zip(y.chunks_exact(8))
        .map(|(x_chunk, y_chunk)| {
            let x_val = u64::from_ne_bytes(x_chunk.try_into().unwrap());
            let y_val = u64::from_ne_bytes(y_chunk.try_into().unwrap());
            (x_val ^ y_val).count_ones()
        })
        .sum::<u32>();

    if x.len() % 8 != 0 {
        distance += x
            .chunks_exact(8)
            .remainder()
            .iter()
            .zip(y.chunks_exact(8).remainder())
            .map(|(x_byte, y_byte)| (x_byte ^ y_byte).count_ones())
            .sum::<u32>();
    }

    distance as u64
}

#[test]
fn all_same_results() {
    let a ="cd8e98b29187133982909fc8b30e39c7b4dca73128ece9cf22ce64eefcf75a3adb0f129b1b00f63a20209e83cb873df707f1af6a4e3558941556b215461a9cbbbce984233c8b8a51e8bd2d1e7f6500caf59fb497440d15365b81e75d3ca4fc9947d5fcb97a0a7b5e44a6b93ee4f622c9b3157991fecac58f364b23f01fd8621e";
    let b = "860e297e5ce51d3bee094b69bedaaf4ec5d74aa639fec1980ac8d6debb77ff8a323350ab4217867a2521d1248f878dc71f39ede3ea357ef39065da261f9ab470ce6884a3e8a6727d1a3c2614ab66481683f63c01de17b4f59d11659ab5a4310121fccc69418839ff6783f9ce7d760ac8e3db7824eef28d0f12fc6b3c1ef8d75c";
    let a = hex::decode(a).unwrap();
    let b = hex::decode(b).unwrap();

    let expected = naive_hamming_distance(&a, &b);

    // Compare with naive_iter implementation
    assert_eq!(expected, naive_hamming_distance_iter(&a, &b));

    // Compare with auto vectorized implementation
    assert_eq!(expected, hamming_distance_auto_vectorized(&a, &b));

    // Compare with hamming crate
    assert_eq!(expected, hamming::distance_fast(&a, &b).unwrap());

    // Compare with hamming_rs crate (x86/x86_64 only)
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    assert_eq!(expected, hamming_rs::distance_faster(&a, &b));

    // Compare with simsimd crate
    assert_eq!(
        expected,
        simsimd::BinarySimilarity::hamming(&a, &b).unwrap() as u64
    );
}

//! Fixed-size array APIs for bitwise Hamming distance.
//!
//! Use this module when the vector size is known at compile time (e.g., 1024-bit
//! embeddings stored as `[u8; 128]`). The compiler can fully unroll and vectorize
//! the loop, yielding the fastest code.

use crate::{array_impl, array_threshold_impl, define_hamming_fn};

define_hamming_fn! {
    /// Compute the bitwise Hamming distance between two fixed-size byte arrays.
    ///
    /// This is the recommended API when the vector size is known at compile time.
    /// The compiler can fully unroll the inner loop and emit optimal SIMD
    /// instructions for the target platform.
    ///
    /// # Example
    ///
    /// ```
    /// use hamming_bitwise_fast::array;
    ///
    /// let a: [u8; 128] = [0x12; 128];  // 1024-bit
    /// let b: [u8; 128] = [0xFE; 128];
    /// let distance = array::distance(&a, &b);
    /// ```
    pub fn distance<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
        array_impl(a, b)
    }
}

define_hamming_fn! {
    /// Compute Hamming distance from one source to many targets (one-to-many).
    ///
    /// Faster than calling [`distance`] in a loop on x86 with `multiversion_x86`,
    /// because CPU dispatch happens once per batch instead of once per comparison.
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != targets.len()`.
    ///
    /// # Example
    ///
    /// ```
    /// use hamming_bitwise_fast::array;
    ///
    /// let source: [u8; 128] = [0; 128];
    /// let targets = vec![[1u8; 128], [2u8; 128], [3u8; 128]];
    /// let mut distances = vec![0u32; 3];  // pre-allocate and reuse
    ///
    /// array::batch(&source, &targets, &mut distances);
    /// ```
    pub fn batch<const N: usize>(
        source: &[u8; N],
        targets: &[[u8; N]],
        out: &mut [u32],
    ) {
        assert_eq!(targets.len(), out.len());

        // Call distance() (the public multiversion function) rather than array_impl
        // directly. The multiversion dispatch creates a boundary that prevents the
        // compiler from seeing the contiguous `&[[u8; N]]` layout, avoiding slow
        // VPGATHERQQ gather instructions on AVX-512. This approach is ~16% faster
        // than using black_box to hide the memory layout.
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            *dist = distance(source, target);
        }
    }
}

define_hamming_fn! {
    /// Compute Hamming distance with early exit when distance exceeds a threshold.
    ///
    /// Returns `Some(distance)` if the distance is `<= max`, or `None` if it
    /// exceeds the threshold. Internally checks the running distance every 512
    /// bits (64 bytes).
    ///
    /// **When to use:** Nearest-neighbor search where most candidates are far
    /// from the query vector (e.g., threshold is ~10% of max possible distance).
    ///
    /// **When NOT to use:** If most comparisons will pass the threshold, use
    /// [`distance`] instead — the early-exit checks add overhead with no benefit.
    ///
    /// # Example
    ///
    /// ```
    /// use hamming_bitwise_fast::array;
    ///
    /// let a: [u8; 128] = [0; 128];
    /// let b: [u8; 128] = [0xFF; 128];  // distance = 1024
    ///
    /// // Tight threshold: exits early without processing full vector
    /// assert_eq!(array::threshold(&a, &b, 100), None);
    ///
    /// // Loose threshold: computes full distance
    /// assert_eq!(array::threshold(&a, &b, 2000), Some(1024));
    /// ```
    pub fn threshold<const N: usize>(
        a: &[u8; N],
        b: &[u8; N],
        max: u32,
    ) -> Option<u32> {
        array_threshold_impl(a, b, max)
    }
}

define_hamming_fn! {
    /// Batch Hamming distance with early exit: compare one source against many
    /// targets, skipping comparisons that exceed `max`.
    ///
    /// Targets that exceed the threshold are written as [`u32::MAX`] in the
    /// output. Returns the minimum distance found (or [`u32::MAX`] if all
    /// targets exceeded the threshold).
    ///
    /// Designed for streaming top-k search: use the heap's worst entry as the
    /// threshold — it tightens as better items are found, causing more early
    /// exits over time.
    ///
    /// ```
    /// use hamming_bitwise_fast::array;
    ///
    /// // Maintain a heap of top-k items. Use the heap's worst
    /// // score as the threshold — it tightens as better items
    /// // are found, causing more early exits over time.
    /// // let threshold = heap.peek().map_or(u32::MAX, |worst| worst.distance);
    /// // let best = array::batch_threshold(&item, &interests, threshold, &mut out);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != targets.len()`.
    ///
    /// # Example
    ///
    /// ```
    /// use hamming_bitwise_fast::array;
    ///
    /// let source: [u8; 128] = [0; 128];
    /// let targets = vec![[0xFFu8; 128], [0u8; 128], [1u8; 128]];
    /// let mut distances = vec![0u32; 3];
    ///
    /// let best = array::batch_threshold(
    ///     &source, &targets, 500, &mut distances,
    /// );
    ///
    /// assert_eq!(distances[0], u32::MAX); // 1024 > 500, rejected
    /// assert_eq!(distances[1], 0);        // within threshold
    /// assert_eq!(distances[2], 128);      // within threshold
    /// assert_eq!(best, 0);                // minimum distance found
    /// ```
    pub fn batch_threshold<const N: usize>(
        source: &[u8; N],
        targets: &[[u8; N]],
        max: u32,
        out: &mut [u32],
    ) -> u32 {
        assert_eq!(targets.len(), out.len());
        let mut best = u32::MAX;
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            match threshold(source, target, max) {
                Some(d) => {
                    *dist = d;
                    if d < best { best = d; }
                }
                None => {
                    *dist = u32::MAX;
                }
            }
        }
        best
    }
}

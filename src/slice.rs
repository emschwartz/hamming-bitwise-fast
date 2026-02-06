//! Variable-length slice APIs for bitwise Hamming distance.
//!
//! Use this module when vector sizes are determined at runtime. When the size
//! is known at compile time, prefer the [`array`](crate::array) module for
//! better performance.

use crate::{define_hamming_fn, slice_impl, slice_threshold_impl};

define_hamming_fn! {
    /// Compute the bitwise Hamming distance between two byte slices.
    ///
    /// # Panics
    ///
    /// Panics if the slices have different lengths.
    ///
    /// # Example
    ///
    /// ```
    /// use hamming_bitwise_fast::slice;
    ///
    /// let a = vec![0xFFu8; 128];
    /// let b = vec![0x00u8; 128];
    /// let distance = slice::distance(&a, &b);  // 1024
    /// ```
    pub fn distance(a: &[u8], b: &[u8]) -> u32 {
        slice_impl(a, b)
    }
}

define_hamming_fn! {
    /// Compute Hamming distance from one source slice to many target slices
    /// (one-to-many).
    ///
    /// Faster than calling [`distance`] in a loop for one-to-many comparisons.
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != targets.len()` or any target has a different
    /// length than `source`.
    ///
    /// # Example
    ///
    /// ```
    /// use hamming_bitwise_fast::slice;
    ///
    /// let source = vec![0u8; 128];
    /// let targets_owned: Vec<Vec<u8>> = vec![vec![1u8; 128], vec![2u8; 128]];
    /// let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();
    /// let mut distances = vec![0u32; 2];  // pre-allocate and reuse
    ///
    /// slice::batch(&source, &targets, &mut distances);
    /// ```
    pub fn batch(source: &[u8], targets: &[&[u8]], out: &mut [u32]) {
        assert_eq!(targets.len(), out.len());

        // For slices, the data layout (&[&[u8]]) isn't contiguous, so gather
        // instructions aren't a concern. Calling slice_impl directly is faster.
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            *dist = slice_impl(source, target);
        }
    }
}

define_hamming_fn! {
    /// Compute Hamming distance with early exit when distance exceeds a threshold.
    ///
    /// Returns `Some(distance)` if the distance is `<= max`, or `None` if it
    /// exceeds the threshold. Internally checks the running distance every 256
    /// bits (32 bytes).
    ///
    /// # Panics
    ///
    /// Panics if the slices have different lengths.
    ///
    /// # Example
    ///
    /// ```
    /// use hamming_bitwise_fast::slice;
    ///
    /// let a = vec![0u8; 128];
    /// let b = vec![0xFFu8; 128];  // distance = 1024
    ///
    /// assert_eq!(slice::threshold(&a, &b, 100), None);
    /// assert_eq!(slice::threshold(&a, &b, 2000), Some(1024));
    /// ```
    pub fn threshold(a: &[u8], b: &[u8], max: u32) -> Option<u32> {
        slice_threshold_impl(a, b, max)
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
    /// # Panics
    ///
    /// Panics if `out.len() != targets.len()` or any target has a different
    /// length than `source`.
    ///
    /// # Example
    ///
    /// ```
    /// use hamming_bitwise_fast::slice;
    ///
    /// let source = vec![0u8; 128];
    /// let targets_owned = vec![vec![0xFFu8; 128], vec![0u8; 128], vec![1u8; 128]];
    /// let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();
    /// let mut distances = vec![0u32; 3];
    ///
    /// let best = slice::batch_threshold(
    ///     &source, &targets, 500, &mut distances,
    /// );
    ///
    /// assert_eq!(distances[0], u32::MAX); // 1024 > 500, rejected
    /// assert_eq!(distances[1], 0);        // within threshold
    /// assert_eq!(distances[2], 128);      // within threshold
    /// assert_eq!(best, 0);                // minimum distance found
    /// ```
    pub fn batch_threshold(source: &[u8], targets: &[&[u8]], max: u32, out: &mut [u32]) -> u32 {
        assert_eq!(targets.len(), out.len());
        let mut best = u32::MAX;
        for (target, dist) in targets.iter().zip(out.iter_mut()) {
            match slice_threshold_impl(source, target, max) {
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

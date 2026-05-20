// Asm-inspection example for investigating gather behavior under LTO.
//
// Build:
//   cargo build --release --example asm_check                    # default (multiversion + lto)
//   RUSTFLAGS="-C target-cpu=native" cargo build --release --example asm_check
//   RUSTFLAGS="-C target-cpu=znver4" cargo build --release --example asm_check
//   cargo build --release --example asm_check --no-default-features
//
// Inspect:
//   objdump --no-show-raw-insn --disassembler-color=on -d target/release/examples/asm_check \
//     | sed -n '/<run_with_barrier>:/,/^[0-9a-f]\+ <[^>]\+>:/p'
//   objdump -d target/release/examples/asm_check | grep -ci 'vpgather\|vpscatter'
//
// What we're hunting for:
//   - `vpgatherqq` / `vpgatherdq` instructions in `run_no_barrier`
//   - whether the barriered version still gets clean `vmovdqu64` + `vpopcntq`
//
// The const N = 32 (256-byte vectors) matches the largest benchmark shape.

use hamming_bitwise_fast::array;

const N: usize = 32; // 256 bytes per vector
const BATCH: usize = 1000;

#[inline(never)]
#[no_mangle]
pub fn run_with_barrier(source: &[u8; N], targets: &[[u8; N]; BATCH], out: &mut [u32; BATCH]) {
    array::batch(source, targets, out);
}

/// Manually inlined batch loop WITHOUT the opaque_ptr barrier.
/// Mirrors the body of `array::batch` exactly, minus the asm! line.
#[inline(never)]
#[no_mangle]
pub fn run_no_barrier(source: &[u8; N], targets: &[[u8; N]; BATCH], out: &mut [u32; BATCH]) {
    assert_eq!(targets.len(), out.len());
    for (target, dist) in targets.iter().zip(out.iter_mut()) {
        // No opaque_ptr barrier.
        let a_chunks = source.chunks_exact(8);
        let b_chunks = target.chunks_exact(8);

        let main: u32 = a_chunks
            .clone()
            .zip(b_chunks.clone())
            .map(|(a, b)| {
                let a = u64::from_ne_bytes(a.try_into().unwrap());
                let b = u64::from_ne_bytes(b.try_into().unwrap());
                (a ^ b).count_ones()
            })
            .sum();

        let rem: u32 = a_chunks
            .remainder()
            .iter()
            .zip(b_chunks.remainder())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum();

        *dist = main + rem;
    }
}

#[inline(never)]
#[no_mangle]
pub fn run_single_distance(a: &[u8; N], b: &[u8; N]) -> u32 {
    array::distance(a, b)
}

fn main() {
    // Touch the symbols so the linker keeps them.
    let source = [0u8; N];
    let targets = Box::new([[0u8; N]; BATCH]);
    let mut out = Box::new([0u32; BATCH]);
    run_with_barrier(&source, &targets, &mut out);
    run_no_barrier(&source, &targets, &mut out);
    let d = run_single_distance(&source, &source);
    println!("done: {} {} {}", out[0], out[BATCH - 1], d);
}

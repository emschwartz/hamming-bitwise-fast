fn main() {
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_features = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let multiversion_enabled = std::env::var("CARGO_FEATURE_MULTIVERSION_X86").is_ok();

    if target_arch != "x86" && target_arch != "x86_64" {
        return;
    }

    // multiversion handles runtime dispatch - no warning needed
    if multiversion_enabled {
        return;
    }

    let features: Vec<&str> = target_features.split(',').collect();
    let has_avx512 = features.iter().any(|f| f.starts_with("avx512"));
    let has_avx2 = features.iter().any(|&f| f == "avx2");

    if has_avx512 {
        // Best performance, no warning needed
    } else if has_avx2 {
        println!(
            "cargo:warning=hamming-bitwise-fast: Building with AVX2 but without AVX-512. \
            For best performance on AVX-512 CPUs, use `--features multiversion_x86` for runtime \
            dispatch or compile with RUSTFLAGS=\"-C target-cpu=native\" on an AVX-512 machine."
        );
    } else {
        println!(
            "cargo:warning=hamming-bitwise-fast: Building for {} without SIMD optimizations. \
            For optimal performance, either: (1) enable `--features multiversion_x86` \
            for runtime CPU dispatch, or (2) compile with RUSTFLAGS=\"-C target-cpu=native\".",
            target_arch
        );
    }
}

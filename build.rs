fn main() {
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let multiversion_enabled = std::env::var("CARGO_FEATURE_MULTIVERSION_X86").is_ok();

    if (target_arch == "x86" || target_arch == "x86_64") && !multiversion_enabled {
        println!(
            "cargo:warning=hamming-bitwise-fast: Building for {} without `multiversion_x86` feature. \
            For optimal performance, either: (1) enable the feature with `--features multiversion_x86` \
            for runtime CPU dispatch, or (2) compile with RUSTFLAGS=\"-C target-cpu=native\" \
            for compile-time optimization.",
            target_arch
        );
    }
}

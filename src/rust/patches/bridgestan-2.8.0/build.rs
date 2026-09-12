use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/pregenerated_bindings.rs");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    std::fs::copy("src/pregenerated_bindings.rs", out_path.join("bindings.rs"))
        .expect("Failed to copy pregenerated bindings");
}

extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/bridgestan.h");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        .header("src/bridgestan.h")
        .opaque_type("model")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .dynamic_library_name("BridgeStan")
        .dynamic_link_require_all(true)
        // On Windows, stddef.h/stdbool.h pull in CRT declarations
        // (e.g. _errno, __security_init_cookie) that are not exported
        // by the Stan model DLL. Restrict bindings to bs_* symbols
        // and the STREAM_CALLBACK type to avoid GetProcAddress failures.
        .allowlist_function("bs_.*")
        .allowlist_var("bs_.*")
        .allowlist_type("STREAM_CALLBACK")
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

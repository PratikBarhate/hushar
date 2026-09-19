// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Build script for `onnxrt-rs`.
//!
//! By default this script does nothing: the crate compiles against the bindings checked
//! in at `src/bindings/ort.rs`, so neither libclang nor an ONNX Runtime installation is
//! needed to build.
//!
//! With `--features bindgen` the bindings are regenerated from the vendored header into
//! `OUT_DIR`. Set `ORT_HEADER_DIR` to bind a different ONNX Runtime release.
//!
//! No linking happens in either mode. `libonnxruntime` is resolved at run time with
//! `libloading`, which keeps this crate buildable on hosts that have no ONNX Runtime at
//! all.

fn main() {
    #[cfg(feature = "bindgen")]
    generate_bindings();

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ORT_HEADER_DIR");
}

/// Regenerates the ONNX Runtime bindings from the vendored header into `OUT_DIR`.
///
/// Only `OrtGetApiBase` plus the `Ort*`/`ONNX*` types are allowlisted, because the C API
/// is reached entirely through the `OrtApi` function-pointer table that function returns.
/// Enums stay plain integer constants so the generated module remains a faithful
/// transcription of the header; the safe layer maps them onto real Rust enums. Layout
/// tests are emitted as `#[test]` size and alignment assertions, so drift between the
/// header and the bindings fails `cargo test`.
///
/// # Panics
///
/// Panics if the header is not found, if bindgen cannot generate the bindings, or if they
/// cannot be written to `OUT_DIR`.
#[cfg(feature = "bindgen")]
fn generate_bindings() {
    use std::path::PathBuf;

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let header_dir = std::env::var("ORT_HEADER_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest_dir.join("headers"));
    let header = header_dir.join("onnxruntime_c_api.h");

    assert!(
        header.is_file(),
        "ONNX Runtime C API header not found at {}. Set ORT_HEADER_DIR to the \
         directory holding onnxruntime_c_api.h.",
        header.display()
    );

    let bindings = bindgen::Builder::default()
        .header(header.to_str().expect("header path is not valid UTF-8"))
        .clang_arg(format!("-I{}", header_dir.display()))
        .allowlist_function("OrtGetApiBase")
        .allowlist_type("Ort.*")
        .allowlist_type("ONNX.*")
        .allowlist_var("ORT_API_VERSION")
        .default_enum_style(bindgen::EnumVariation::Consts)
        .derive_debug(true)
        .derive_default(false)
        .layout_tests(true)
        .generate_comments(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("failed to generate ONNX Runtime bindings");

    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(&out)
        .expect("failed to write ONNX Runtime bindings");
    println!(
        "cargo:warning=regenerated ONNX Runtime bindings at {}",
        out.display()
    );
}

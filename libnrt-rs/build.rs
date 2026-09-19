// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Build script for `libnrt-rs`.
//!
//! By default this does nothing and the crate compiles against the bindings checked in
//! at `src/bindings/nrt.rs`, so the Neuron SDK is not needed at build time. With
//! `--features bindgen` the bindings are regenerated from the SDK headers found under
//! `NEURON_INCLUDE_DIR` (default `/opt/aws/neuron/include`).
//!
//! No linking happens here. `libnrt.so.1` is opened at run time with `libloading`, so a
//! binary built from this crate runs unchanged on hosts with and without Neuron
//! hardware; it fails at `Runtime::init` on the latter.

/// Where the Neuron SDK installs its headers.
const DEFAULT_INCLUDE_DIR: &str = "/opt/aws/neuron/include";

/// Emits the rerun triggers, and regenerates the bindings under `--features bindgen`.
///
/// [`DEFAULT_INCLUDE_DIR`] is discarded at the end so it does not warn as unused when
/// the feature is off.
fn main() {
    #[cfg(feature = "bindgen")]
    generate_bindings();

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=headers/wrapper.h");
    println!("cargo:rerun-if-env-changed=NEURON_INCLUDE_DIR");

    let _ = DEFAULT_INCLUDE_DIR;
}

/// Regenerates the Neuron runtime bindings into `$OUT_DIR/bindings.rs`.
///
/// Output is restricted to the `nrt_*` and `NRT_*` surface this crate wraps. The header
/// search adds the SDK include directory with `-I` and `headers/shim` with `-idirafter`,
/// which is searched after every other include path, so a real Linux host picks up its
/// own `linux/types.h` and only a non-Linux host falls back to the shim; see that file
/// for why the fallback is needed and why it is safe.
///
/// # Panics
///
/// If `nrt/nrt.h` is not under `NEURON_INCLUDE_DIR`, or if generating or writing the
/// bindings fails.
#[cfg(feature = "bindgen")]
fn generate_bindings() {
    use std::path::PathBuf;

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let include_dir =
        std::env::var("NEURON_INCLUDE_DIR").unwrap_or_else(|_| DEFAULT_INCLUDE_DIR.to_owned());
    let include_dir = PathBuf::from(include_dir);

    assert!(
        include_dir.join("nrt/nrt.h").is_file(),
        "Neuron SDK headers not found: {} does not contain nrt/nrt.h. Install \
         aws-neuronx-runtime-lib, or set NEURON_INCLUDE_DIR to the SDK include \
         directory.",
        include_dir.display()
    );

    let wrapper = manifest_dir.join("headers/wrapper.h");
    let shim_dir = manifest_dir.join("headers/shim");

    let bindings = bindgen::Builder::default()
        .header(wrapper.to_str().expect("wrapper path is not valid UTF-8"))
        .clang_arg(format!("-I{}", include_dir.display()))
        .clang_arg(format!("-idirafter{}", shim_dir.display()))
        .allowlist_function("nrt_.*")
        .allowlist_type("nrt_.*")
        .allowlist_type("NRT_.*")
        .allowlist_var("NRT_.*")
        .default_enum_style(bindgen::EnumVariation::Consts)
        .derive_debug(true)
        .derive_default(false)
        .layout_tests(true)
        .generate_comments(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("failed to generate Neuron runtime bindings");

    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(&out)
        .expect("failed to write Neuron runtime bindings");
    println!(
        "cargo:warning=regenerated Neuron runtime bindings at {}",
        out.display()
    );
}

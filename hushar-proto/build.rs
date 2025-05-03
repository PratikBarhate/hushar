//! Build script for compiling protobuf definitions.
//!
//! This script handles the discovery and compilation of `.proto` files
//! using tonic_build, enabling gRPC service generation for the project.

use std::fs;
use std::path::Path;

/// Main build function that discovers and compiles protobuf files.
///
/// This function:
/// 1. Locates all `.proto` files in the "proto" directory
/// 2. Configures tonic_build to generate both server and client code
/// 3. Compiles the discovered protos
/// 4. Sets up cargo to rerun the build when proto files change
///
/// # Errors
///
/// Returns an error if the proto files can't be found or compiled.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_dir = "../schemas/protos";
    let proto_files = find_proto_files(proto_dir)?;

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&proto_files, &[proto_dir])?;

    for proto_file in &proto_files {
        println!("cargo:info=Compiling proto file: {}", proto_file);
    }
    for proto_file in &proto_files {
        println!("cargo:rerun-if-changed={}", proto_file);
    }
    Ok(())
}

/// Recursively finds all `.proto` files in a directory and its subdirectories.
///
/// # Arguments
///
/// * `dir` - The path to the directory to search in
///
/// # Returns
///
/// A vector of strings containing the paths to all found proto files.
///
/// # Errors
///
/// Returns an error if the directory cannot be read or if there are issues
/// with file system operations.
fn find_proto_files<P: AsRef<Path>>(dir: P) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut proto_files = Vec::new();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("proto") {
            if let Some(path_str) = path.to_str() {
                proto_files.push(path_str.to_string());
            }
        } else if path.is_dir() {
            let mut sub_protos = find_proto_files(path)?;
            proto_files.append(&mut sub_protos);
        }
    }

    Ok(proto_files)
}

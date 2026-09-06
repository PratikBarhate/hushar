// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Helpers shared by the integration tests.
//!
//! Every test here needs a real `libonnxruntime`, so all of them skip with a
//! stated reason when one cannot be loaded rather than failing. That keeps the
//! suite green on machines and CI images without ONNX Runtime installed, and
//! `runtime_is_available_when_configured` in `onnxruntime_smoke.rs` guards
//! against a skip hiding a genuine breakage.

#![allow(dead_code)] // Each test binary uses a different subset.

use std::path::PathBuf;
use std::sync::Arc;

use onnxrt_rs::{Api, Environment, ExecutionProvider, Session, SessionBuilder};

/// Loads the runtime, or returns `None` after explaining the skip.
pub fn api() -> Option<Api> {
    match Api::load() {
        Ok(api) => Some(api),
        Err(e) => {
            eprintln!("skipping: no ONNX Runtime available ({e})");
            None
        }
    }
}

/// Reads a fixture from this crate's own `test-data`.
///
/// This crate's, not the sibling service's: these tests must need nothing but
/// `onnxrt-rs` and a `libonnxruntime`, so the suite still runs when the crate is
/// taken out of this workspace. `sigmoid_model_3.onnx` and `mixed_io.onnx` are
/// therefore copies of fixtures hushar also tests, but neither is copied by hand
/// -- `scripts/generate_test_model.py` and `scripts/generate_typed_test_models.py`
/// write every copy from one definition, which is what keeps them from drifting.
pub fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test-data")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "could not read {}: {e}. Regenerate the fixtures with the scripts in \
             `scripts/`, from the repository root",
            path.display()
        )
    })
}

/// The 3-feature / 2-output sigmoid model, which most of the suite runs.
pub fn test_model() -> Vec<u8> {
    fixture("sigmoid_model_3.onnx")
}

/// One row of features for the fixture, and the shape describing it.
pub fn single_row() -> (Vec<f32>, [i64; 2]) {
    (vec![1.0, 2.0, 3.0], [1, 3])
}

/// The process-wide environment.
///
/// Shared, not per-call: ONNX Runtime allows only one environment using the
/// default logger at a time, and these tests run in parallel.
pub fn environment() -> Arc<Environment> {
    Environment::shared("onnxrt-rs-test").expect("shared environment")
}

/// Builds a session on `provider`.
pub fn session_on(provider: &ExecutionProvider) -> onnxrt_rs::Result<Session> {
    SessionBuilder::new(&environment())?
        .intra_op_threads(1)?
        .execution_provider(provider)?
        .build_from_memory(&test_model())
}

/// Runs the fixture's single row and returns the scores.
pub fn scores_on(provider: &ExecutionProvider) -> onnxrt_rs::Result<Vec<f32>> {
    let session = session_on(provider)?;
    let (mut features, shape) = single_row();
    let output = session.run_single(&mut features, &shape)?;
    Ok(output.as_slice::<f32>()?.to_vec())
}

/// The CPU result, which every accelerator is compared against.
///
/// CPU is always present, so this is a usable reference on any host.
pub fn cpu_scores() -> Vec<f32> {
    scores_on(&ExecutionProvider::Cpu).expect("CPU inference must work")
}

/// Asserts an accelerator agrees with the CPU baseline.
///
/// Accelerators legitimately differ in the last bits — different kernels,
/// different orders of accumulation, sometimes reduced precision — so this is a
/// tolerance rather than equality. It is still tight enough to catch a genuinely
/// wrong result, which is the failure worth catching: a provider that silently
/// computes something else is far worse than one that refuses to load.
pub fn assert_matches_cpu(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label} returned {} scores, CPU returned {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-4,
            "{label} score {i} was {a}, CPU gave {e} (all: {actual:?} vs {expected:?})"
        );
    }
}

/// Asserts every score is a plausible sigmoid output.
pub fn assert_sigmoid_range(label: &str, scores: &[f32]) {
    assert_eq!(scores.len(), 2, "{label}: fixture emits 2 scores");
    for s in scores {
        assert!(
            *s > 0.0 && *s < 1.0,
            "{label}: sigmoid output out of range: {s} (all: {scores:?})"
        );
    }
}

/// Runs a provider's test body, skipping with a reason when the hardware or the
/// library cannot support it.
///
/// The three outcomes are deliberately distinguished, because they mean very
/// different things:
///
/// * no runtime at all — skip
/// * runtime present but built without this provider — skip, and say so
/// * runtime present and provider present — run, and a failure is a real failure
pub fn with_provider(provider: ExecutionProvider, body: impl FnOnce(&ExecutionProvider)) {
    let Some(api) = api() else { return };

    if !provider.is_enabled() {
        eprintln!(
            "skipping {}: this build does not enable the `{}` feature",
            provider.name(),
            provider.required_feature().unwrap_or("<none>")
        );
        return;
    }

    match provider.is_available(api) {
        Ok(true) => body(&provider),
        Ok(false) => eprintln!(
            "skipping {}: this ONNX Runtime build has no such provider (it has: {})",
            provider.name(),
            api.available_providers().unwrap_or_default().join(", ")
        ),
        Err(e) => panic!("could not query available providers: {e}"),
    }
}

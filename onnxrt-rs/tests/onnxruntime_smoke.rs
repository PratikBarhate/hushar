// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! End-to-end tests against a real ONNX Runtime shared library.
//!
//! These need `libonnxruntime` at run time. Point `ORT_DYLIB_PATH` at it, or
//! install it somewhere the dynamic loader searches:
//!
//! ```text
//! ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib cargo test -p onnxrt-rs
//! ```
//!
//! When no runtime can be loaded each test reports that it was skipped and
//! passes, so the suite stays green on machines and CI images without ONNX
//! Runtime. `runtime_is_available_when_configured` guards against that skip
//! silently hiding a real breakage.

use std::sync::Arc;

use onnxrt_rs::{Api, Environment, Error, SessionBuilder};

mod common;

use common::{api, test_model};

/// A session on the default (CPU) provider, plus the environment it belongs to.
fn session(_api: Api) -> (Arc<Environment>, onnxrt_rs::Session) {
    let env = common::environment();
    let session = SessionBuilder::new(&env)
        .expect("session builder")
        .intra_op_threads(1)
        .expect("intra op threads")
        .build_from_memory(&test_model())
        .expect("session");
    (env, session)
}

/// Asserts the runtime loads and reports a version when `ORT_DYLIB_PATH` is set.
///
/// A runtime the caller named explicitly must load, so this failing is a bug rather
/// than a reason to skip.
#[test]
fn runtime_is_available_when_configured() {
    if std::env::var_os(onnxrt_rs::DYLIB_PATH_ENV).is_some() {
        let api = Api::load().expect("ORT_DYLIB_PATH is set, so the runtime must load");
        assert!(
            !api.version().is_empty(),
            "runtime should report a version string"
        );
        eprintln!("loaded ONNX Runtime build info: {}", api.version());
    }
}

/// Asserts a single row returns a `[1, 2]` tensor of scores strictly inside (0, 1).
#[test]
fn single_row_inference_produces_sigmoid_outputs() {
    let Some(api) = api() else { return };
    let (_env, session) = session(api);

    let mut features = vec![1.0f32, 2.0, 3.0];
    let output = session
        .run_single(&mut features, &[1, 3])
        .expect("inference should succeed");

    assert_eq!(output.shape().expect("shape"), vec![1, 2]);
    let scores = output.as_slice::<f32>().expect("f32 output");
    assert_eq!(scores.len(), 2);
    for s in scores {
        assert!(
            *s > 0.0 && *s < 1.0,
            "sigmoid output {s} should be strictly inside (0, 1)"
        );
    }
}

/// Asserts a batch of 3 rows x 3 features, laid out row-major, returns a `[3, 2]`
/// tensor of 6 scores.
#[test]
fn batch_inference_returns_one_row_of_scores_per_input_row() {
    let Some(api) = api() else { return };
    let (_env, session) = session(api);

    let mut features = vec![1.0f32, 2.0, 3.0, 0.5, 1.0, 1.5, 0.1, 0.2, 0.3];
    let output = session
        .run_single(&mut features, &[3, 3])
        .expect("batch inference should succeed");

    assert_eq!(output.shape().expect("shape"), vec![3, 2]);
    assert_eq!(output.len().expect("len"), 6);

    let scores = output.as_slice::<f32>().expect("f32 output");
    assert_eq!(scores.len(), 6);
    assert!(scores.iter().all(|s| *s >= 0.0 && *s <= 1.0));
}

/// Asserts the fixture reports one input and one output, each with a non-empty name.
///
/// The names come from the model, so only non-emptiness is asserted rather than
/// whatever the exporter chose.
#[test]
fn model_io_names_are_discovered() {
    let Some(api) = api() else { return };
    let (_env, session) = session(api);

    assert_eq!(
        session.input_names().len(),
        1,
        "the sigmoid test model has a single input"
    );
    assert_eq!(
        session.output_names().len(),
        1,
        "the sigmoid test model has a single output"
    );
    assert!(session.input_names().iter().all(|n| !n.is_empty()));
    assert!(session.output_names().iter().all(|n| !n.is_empty()));
}

/// Asserts garbage model bytes surface as an `Error::Ort` status rather than loading.
#[test]
fn invalid_model_bytes_are_reported_as_an_ort_error() {
    let Some(api) = api() else { return };
    let env = Environment::with_api(api, "hushar-bad-model", onnxrt_rs::LogLevel::Fatal)
        .expect("environment");

    let err = SessionBuilder::new(&env)
        .expect("session builder")
        .build_from_memory(&[0u8, 1, 2, 3])
        .expect_err("garbage bytes must not load as a model");

    assert!(
        matches!(err, Error::Ort { .. }),
        "expected an Ort status error, got {err:?}"
    );
}

/// Asserts a shape whose element count disagrees with the buffer — 3 values as a 2x3
/// tensor — is rejected as `ShapeMismatch` before reaching C.
#[test]
fn shape_that_disagrees_with_the_buffer_is_rejected_before_reaching_c() {
    let Some(api) = api() else { return };
    let (_env, session) = session(api);

    let mut features = vec![1.0f32, 2.0, 3.0];
    let err = session
        .run_single(&mut features, &[2, 3])
        .expect_err("mismatched shape must be rejected");

    match err {
        Error::ShapeMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, 6);
            assert_eq!(actual, 3);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

/// Asserts reading a float output as `i64` is an `ElementTypeMismatch`.
#[test]
fn reading_an_output_as_the_wrong_element_type_is_an_error() {
    let Some(api) = api() else { return };
    let (_env, session) = session(api);

    let mut features = vec![1.0f32, 2.0, 3.0];
    let output = session
        .run_single(&mut features, &[1, 3])
        .expect("inference");

    let err = output
        .as_slice::<i64>()
        .expect_err("a float tensor must not be readable as i64");
    assert!(
        matches!(err, Error::ElementTypeMismatch { .. }),
        "expected ElementTypeMismatch, got {err:?}"
    );
}

/// Asserts the available-provider list always names CPU, which every build has.
#[test]
fn available_providers_always_include_cpu() {
    let Some(api) = api() else { return };
    let providers = api.available_providers().expect("query providers");
    eprintln!("this ONNX Runtime was built with: {}", providers.join(", "));
    assert!(
        providers.iter().any(|p| p.contains("CPU")),
        "CPU is built into every ONNX Runtime, but got: {providers:?}"
    );
}

/// Asserts the declared input and output shapes are readable: 3 features in, 2
/// sigmoid outputs.
///
/// Only the trailing dimension is asserted, because the batch axis may be concrete or
/// dynamic (-1) depending on how the model was exported.
#[test]
fn declared_input_and_output_shapes_are_readable() {
    let Some(api) = api() else { return };
    let (_env, session) = session(api);

    let input = session.input_shape(0).expect("input shape");
    let output = session.output_shape(0).expect("output shape");
    eprintln!("model shapes: input {input:?} output {output:?}");

    assert_eq!(input.len(), 2, "expected a 2-D input, got {input:?}");
    assert_eq!(input[1], 3, "expected 3 features, got {input:?}");
    assert_eq!(output.len(), 2, "expected a 2-D output, got {output:?}");
    assert_eq!(output[1], 2, "expected 2 outputs, got {output:?}");
}

/// Asserts an out-of-range shape index is an `IndexOutOfRange` naming index and count.
#[test]
fn out_of_range_shape_index_is_rejected() {
    let Some(api) = api() else { return };
    let (_env, session) = session(api);
    match session.input_shape(99) {
        Err(Error::IndexOutOfRange { index, count, .. }) => {
            assert_eq!(index, 99);
            assert_eq!(count, 1);
        }
        other => panic!("expected IndexOutOfRange, got {other:?}"),
    }
}

/// Asserts requesting a provider the loaded library lacks is a `ProviderUnavailable`
/// naming what is available, not a bare ONNX Runtime status.
#[test]
fn requesting_an_absent_provider_names_what_is_available() {
    let Some(api) = api() else { return };
    let env = Environment::shared("onnxrt-rs-absent-provider").expect("create env");

    let bogus = onnxrt_rs::ExecutionProvider::Custom {
        name: "NoSuchAccelerator".to_owned(),
        options: Vec::new(),
    };

    match SessionBuilder::new(&env)
        .expect("builder")
        .execution_provider(&bogus)
    {
        Err(Error::ProviderUnavailable {
            provider,
            available,
        }) => {
            assert_eq!(provider, "NoSuchAccelerator");
            assert!(
                available.contains("CPU"),
                "the available list should name real providers, got {available:?}"
            );
        }
        Ok(_) => panic!("a nonexistent provider must not be accepted"),
        Err(e) => panic!("expected ProviderUnavailable, got {e}"),
    }
    let _ = api;
}

/// Asserts [`Environment::shared`] hands out one environment per process, including
/// under concurrent calls from several threads.
///
/// ONNX Runtime permits only one environment using the default logger at a time, and
/// creating or tearing down environments concurrently fails with "Only one instance of
/// LoggingManager created with InstanceType::Default can exist at any point in time".
/// `shared` therefore creates exactly one environment per process and never drops it.
#[test]
fn shared_environment_is_created_once() {
    let Some(_api) = api() else { return };

    let first = Environment::shared("onnxrt-rs-shared-test").expect("first");
    let second = Environment::shared("a-different-name-that-is-ignored").expect("second");
    assert!(
        Arc::ptr_eq(&first, &second),
        "shared() must hand out one environment, not one per call"
    );

    let handles: Vec<_> = (0..8)
        .map(|_| std::thread::spawn(|| Environment::shared("concurrent").expect("shared")))
        .collect();
    let environments: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().expect("join"))
        .collect();
    for env in &environments {
        assert!(
            Arc::ptr_eq(env, &first),
            "every thread must observe the same environment"
        );
    }
}

/// Asserts an `Arc<Session>` can run inference from several threads at once, which is
/// how hushar serves concurrent requests.
#[test]
fn sessions_can_be_shared_across_threads() {
    let Some(api) = api() else { return };
    let (_env, session) = session(api);
    let session = Arc::new(session);

    let handles: Vec<_> = (0..8)
        .map(|i| {
            let session = Arc::clone(&session);
            std::thread::spawn(move || {
                let mut features = vec![i as f32, 2.0, 3.0];
                let output = session
                    .run_single(&mut features, &[1, 3])
                    .expect("concurrent inference");
                output.to_vec::<f32>().expect("f32 output")
            })
        })
        .collect();

    for handle in handles {
        let scores = handle.join().expect("worker thread panicked");
        assert_eq!(scores.len(), 2);
        assert!(scores.iter().all(|s| *s > 0.0 && *s < 1.0));
    }
}

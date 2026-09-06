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

#[test]
fn runtime_is_available_when_configured() {
    // If the caller explicitly pointed us at a runtime, failing to load it is a
    // real bug, not a reason to skip.
    if std::env::var_os(onnxrt_rs::DYLIB_PATH_ENV).is_some() {
        let api = Api::load().expect("ORT_DYLIB_PATH is set, so the runtime must load");
        assert!(
            !api.version().is_empty(),
            "runtime should report a version string"
        );
        eprintln!("loaded ONNX Runtime build info: {}", api.version());
    }
}

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

#[test]
fn batch_inference_returns_one_row_of_scores_per_input_row() {
    let Some(api) = api() else { return };
    let (_env, session) = session(api);

    // 3 rows x 3 features, laid out row-major.
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
    // Names come from the model, so assert they are non-empty rather than
    // hard-coding whatever the exporter chose.
    assert!(session.input_names().iter().all(|n| !n.is_empty()));
    assert!(session.output_names().iter().all(|n| !n.is_empty()));
}

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

#[test]
fn shape_that_disagrees_with_the_buffer_is_rejected_before_reaching_c() {
    let Some(api) = api() else { return };
    let (_env, session) = session(api);

    // 3 values cannot be a 2x3 tensor.
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

#[test]
fn declared_input_and_output_shapes_are_readable() {
    let Some(api) = api() else { return };
    let (_env, session) = session(api);

    let input = session.input_shape(0).expect("input shape");
    let output = session.output_shape(0).expect("output shape");
    eprintln!("model shapes: input {input:?} output {output:?}");

    // The fixture takes 3 features and produces 2 sigmoid outputs. The batch axis
    // may be concrete or dynamic (-1) depending on how the model was exported, so
    // only the trailing feature dimension is asserted.
    assert_eq!(input.len(), 2, "expected a 2-D input, got {input:?}");
    assert_eq!(input[1], 3, "expected 3 features, got {input:?}");
    assert_eq!(output.len(), 2, "expected a 2-D output, got {output:?}");
    assert_eq!(output[1], 2, "expected 2 outputs, got {output:?}");
}

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

/// Requesting a provider the loaded library lacks must say so clearly, and name
/// what is available, rather than surface a bare ONNX Runtime status.
#[test]
fn requesting_an_absent_provider_names_what_is_available() {
    let Some(api) = api() else { return };
    let env = Environment::shared("onnxrt-rs-absent-provider").expect("create env");

    // A provider name no ONNX Runtime build has.
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

/// The invariant that keeps ONNX Runtime's single-environment rule satisfied.
///
/// ONNX Runtime permits only one environment using the default logger at a time,
/// and creating or tearing down environments concurrently fails with "Only one
/// instance of LoggingManager created with InstanceType::Default can exist at any
/// point in time". [`Environment::shared`] answers that by creating exactly one
/// environment per process and never dropping it, so there is no churn to race.
///
/// This is the guard for that: if `shared` ever started handing out distinct
/// environments, the intermittent failure would come back.
#[test]
fn shared_environment_is_created_once() {
    let Some(_api) = api() else { return };

    let first = Environment::shared("onnxrt-rs-shared-test").expect("first");
    let second = Environment::shared("a-different-name-that-is-ignored").expect("second");
    assert!(
        Arc::ptr_eq(&first, &second),
        "shared() must hand out one environment, not one per call"
    );

    // Also under contention, which is when the original defect showed up.
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

#[test]
fn sessions_can_be_shared_across_threads() {
    let Some(api) = api() else { return };
    let (_env, session) = session(api);
    let session = Arc::new(session);

    // The point of this test is that `Arc<Session>` is usable from several
    // threads at once, which is how hushar serves concurrent requests.
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
